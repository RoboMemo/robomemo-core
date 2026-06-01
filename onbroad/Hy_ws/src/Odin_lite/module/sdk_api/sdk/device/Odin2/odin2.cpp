#include "odin2.h"

#include "cJSON.h"  

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ITransport.hpp"
#include "TcpTransport.hpp"
#include "OdinProtocol.hpp"
#include "http_client.h"
#include "logger.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace odin {
namespace sdk {

namespace {

// Odin2 Command IDs (internal to device implementation)
enum class Odin2CmdId : uint16_t {
  kDeviceQuery = 0x01,
  kVersionQuery = 0x02,
  kCmdIdQueryCapability = 0x03,
  kSetMode = 0x04,
  kSensorMode = 0x05,
  kChannelConfig = 0x06,
  kHeartbeat = 0x07,
};

// TLV Type IDs for ChannelConfig command
enum class ChannelConfigTlvType : uint8_t {
  kDstPort = 0x01,
  kTransportMode = 0x02,
  kResolutionId = 0x03,
  kFpsId = 0x04,
  kFormat = 0x05,
};

bool ParseIpv4Address(const std::string& ip_str, uint32_t* out_host_order) {
  if (!out_host_order) return false;
  in_addr addr;
  if (inet_pton(AF_INET, ip_str.c_str(), &addr) != 1) return false;
  *out_host_order = ntohl(addr.s_addr);
  return true;
}

#ifdef _WIN32
std::atomic<int> g_winsock_ref_count{0};

bool AcquireWinsock() {
  if (g_winsock_ref_count.fetch_add(1) == 0) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
      g_winsock_ref_count.fetch_sub(1);
      return false;
    }
  }
  return true;
}

void ReleaseWinsock() {
  if (g_winsock_ref_count.fetch_sub(1) == 1) {
    WSACleanup();
  }
}
#else
bool AcquireWinsock() { return true; }
void ReleaseWinsock() {}
#endif

}  // namespace

Odin2Device::Odin2Device(OdinDeviceHandle handle)
    : handle_(handle), slam_odom_sync_(new SlamOdomSynchronizer()) {
  // Set SLAM transform function for coordinate transformation using Odom pose
  if (slam_odom_sync_) {
    slam_odom_sync_->SetSlamTransformFunction(
        [](OdinPointCloudPacket& packet, const OdinOdomPacket& odom_pkt) {
          // Match found - transform points to world frame: pt_world = R * pt_imu + t
          const OdinOdomPacket& odom_packet = odom_pkt;

          if (odom_packet.payload.size() < sizeof(OdinOdomData)) {
            return;
          }
          const OdinOdomData* odom_data =
              reinterpret_cast<const OdinOdomData*>(odom_packet.payload.data());

          // Extract quaternion and position from odom
          constexpr double kPosScale = 1e-6;
          constexpr double kOrientScale = 1e-6;
          double tx = odom_data->pos[0] * kPosScale;
          double ty = odom_data->pos[1] * kPosScale;
          double tz = odom_data->pos[2] * kPosScale;
          double qx = odom_data->orient[0] * kOrientScale;
          double qy = odom_data->orient[1] * kOrientScale;
          double qz = odom_data->orient[2] * kOrientScale;
          double qw = odom_data->orient[3] * kOrientScale;

          // Build rotation matrix R^{w}_{i} from quaternion using Eigen
          Eigen::Quaterniond q(qw, qx, qy, qz);
          q.normalize();
          Eigen::Matrix3d R = q.toRotationMatrix();
          Eigen::Vector3d t(tx, ty, tz);

          // Transform each point: pt_world = R * pt_imu + t
          // Read from uint16_t format
          size_t src_point_size = sizeof(OdinSlamPoint<uint16_t>);
          size_t point_count = packet.payload.size() / src_point_size;
          const OdinSlamPoint<uint16_t>* src_points =
              reinterpret_cast<const OdinSlamPoint<uint16_t>*>(packet.payload.data());

          // Create new payload with float xyz
          size_t dst_point_size = sizeof(OdinSlamPoint<float>);
          std::vector<uint8_t> new_payload(point_count * dst_point_size);
          OdinSlamPoint<float>* dst_points =
              reinterpret_cast<OdinSlamPoint<float>*>(new_payload.data());

          for (size_t i = 0; i < point_count; ++i) {
            // Step 1: Convert from uint16_t to float with offset (in meters)
            Eigen::Vector3d pt_imu(static_cast<double>(src_points[i].x) * 0.001,
                                   (static_cast<double>(src_points[i].y) - 30000.0) * 0.001,
                                   (static_cast<double>(src_points[i].z) - 30000.0) * 0.001);

            // Step 2: Apply rotation and translation: pt_world = R * pt_imu + t
            Eigen::Vector3d pt_world = R * pt_imu + t;

            // Step 3: Store as float directly
            dst_points[i].x = static_cast<float>(pt_world.x()) / 0.001f;
            dst_points[i].y = static_cast<float>(pt_world.y()) / 0.001f;
            dst_points[i].z = static_cast<float>(pt_world.z()) / 0.001f;
            dst_points[i].r = src_points[i].r;
            dst_points[i].g = src_points[i].g;
            dst_points[i].b = src_points[i].b;
            dst_points[i].a = src_points[i].a;
          }

          // Update packet with transformed data
          packet.payload = std::move(new_payload);
        });
  }
}

Odin2Device::~Odin2Device() { Disconnect(); }

bool Odin2Device::IsPortAvailable(const std::string& ip, uint16_t port) {
  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) return false;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (ip.empty() || ip == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else {
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
  }

  int result = bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
#ifdef _WIN32
  closesocket(sock);
#else
  close(sock);
#endif
  return result == 0;
}

// Global port tracking for multi-device support
static std::mutex g_port_mutex;
static std::set<uint16_t> g_used_ports;

uint16_t Odin2Device::GenerateAvailablePort() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dist(10000, 60000);

  std::lock_guard<std::mutex> lock(g_port_mutex);
  for (int i = 0; i < 1000; ++i) {
    uint16_t port = dist(gen);
    if (g_used_ports.count(port) == 0 && IsPortAvailable(host_ip_, port)) {
      g_used_ports.insert(port);
      return port;
    }
  }
  return 0;
}

bool Odin2Device::Connect(const DiscoveredDevice& discovered_device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (connected_.load()) return true;

  // Store device info from discovery
  discovered_device_ = discovered_device;  // Save full device info for heartbeat failure
  device_ip_ = discovered_device.ip;
  host_ip_ = discovered_device.host_ip.empty() ? "0.0.0.0" : discovered_device.host_ip;
  serial_number_ = discovered_device.sn;
  model_ = discovered_device.model;
  firmware_version_ = discovered_device.firmware_version;

  if (device_ip_.empty()) {
    LOG_ERROR("Invalid device IP\n");
    return false;
  }

  if (!AcquireWinsock()) {
    LOG_ERROR("AcquireWinsock failed\n");
    return false;
  }

  // ==========================================================================
  // Step 1: Try to establish heartbeat channel (TCP 60002) - optional for legacy devices
  // ==========================================================================
  NetworkAddress local_addr;
  local_addr.ip = host_ip_;
  local_addr.port = 0;  // Let OS assign available port

  // Reset heartbeat support flag
  heartbeat_supported_ = true;
  heartbeat_probing_ = false;  // No need to probe, we test during connect

  std::unique_ptr<ITransport> hb_transport(new TcpTransport());
  NetworkAddress hb_remote_addr;
  hb_remote_addr.ip = device_ip_;
  hb_remote_addr.port = 60002;
  hb_transport->SetRemoteTarget(hb_remote_addr);

  if (!hb_transport->Open(local_addr)) {
    LOG_WARN("Heartbeat transport Open failed (port 60002) - device may not support heartbeat\n");
    heartbeat_supported_ = false;
    // Continue without heartbeat channel
  } else {
    std::unique_ptr<IProtocol> hb_protocol(new OdinProtocolV1());
    heartbeat_channel_.reset(new CommandChannel(std::move(hb_protocol), std::move(hb_transport)));

    CommandChannelConfig hb_cfg;
    hb_cfg.default_timeout_ms = 2000;
    hb_cfg.device_handle = handle_;

    if (!heartbeat_channel_->Start(hb_cfg)) {
      LOG_WARN("Heartbeat CommandChannel start failed - device may not support heartbeat\n");
      heartbeat_channel_.reset();
      heartbeat_supported_ = false;
    } else {
      // Send first heartbeat and wait for response to verify connection
      std::vector<uint8_t> hb_payload;
      OdinCommandSyncResponse hb_response;
      if (!heartbeat_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kHeartbeat), hb_payload,
                                         hb_response, 2000)) {
        LOG_WARN("First heartbeat failed - device may not support heartbeat\n");
        heartbeat_channel_->Stop();
        heartbeat_channel_.reset();
        heartbeat_supported_ = false;
      } else {
        LOG_INFO("Heartbeat channel established (port 60002)\n");
      }
    }
  }

  if (!heartbeat_supported_) {
    LOG_INFO("Device does not support heartbeat, continuing without heartbeat monitoring\n");
  }

  // ==========================================================================
  // Step 2: Establish command channel (TCP 60001)
  // ==========================================================================
  std::unique_ptr<ITransport> cmd_transport(new TcpTransport());
  NetworkAddress cmd_remote_addr;
  cmd_remote_addr.ip = device_ip_;
  cmd_remote_addr.port = 60001;
  cmd_transport->SetRemoteTarget(cmd_remote_addr);

  if (!cmd_transport->Open(local_addr)) {
    LOG_ERROR("Command transport Open failed (port 60001), host_ip[%s]\n", host_ip_.c_str());
    if (heartbeat_channel_) {
      heartbeat_channel_->Stop();
      heartbeat_channel_.reset();
    }
    ReleaseWinsock();
    return false;
  }

  std::unique_ptr<IProtocol> cmd_protocol(new OdinProtocolV1());
  command_channel_.reset(new CommandChannel(std::move(cmd_protocol), std::move(cmd_transport)));

  CommandChannelConfig cmd_cfg;
  cmd_cfg.default_timeout_ms = 3000;
  cmd_cfg.device_handle = handle_;

  if (!command_channel_->Start(cmd_cfg)) {
    LOG_ERROR("Command CommandChannel start failed\n");
    command_channel_.reset();
    if (heartbeat_channel_) {
      heartbeat_channel_->Stop();
      heartbeat_channel_.reset();
    }
    ReleaseWinsock();
    return false;
  }

  if (!ParseIpv4Address(device_ip_, &device_ip_host_)) {
    LOG_ERROR("ParseIpv4Address failed\n");
    command_channel_->Stop();
    command_channel_.reset();
    if (heartbeat_channel_) {
      heartbeat_channel_->Stop();
      heartbeat_channel_.reset();
    }
    ReleaseWinsock();
    return false;
  }

  connected_.store(true);
  LOG_INFO("Odin2Device connected: %s (SN: %s, Model: %s)\n", device_ip_.c_str(),
           serial_number_.c_str(), model_.c_str());
  
  // Start heartbeat thread for connection monitoring
  StartHeartbeat();
  return true;
}

void Odin2Device::Disconnect() {
  // Stop heartbeat first (outside lock to avoid deadlock)
  StopHeartbeat();
  
  std::lock_guard<std::mutex> lock(mutex_);
  if (!connected_.load()) return;

  if (command_channel_) {
    command_channel_->Stop();
    command_channel_.reset();
  }

  if (heartbeat_channel_) {
    heartbeat_channel_->Stop();
    heartbeat_channel_.reset();
  }

  ReleaseWinsock();
  connected_.store(false);
  LOG_INFO("Odin2Device disconnected: %s\n", device_ip_.c_str());
}

bool Odin2Device::IsConnected() const { return connected_.load(); }

OdinDeviceHandle Odin2Device::GetHandle() const { return handle_; }

std::string Odin2Device::GetSerialNumber() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return serial_number_;
}

std::string Odin2Device::GetModel() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return model_;
}

MTConnectionType Odin2Device::GetConnectionType() const { return MTConnectionType::kEthernet; }

const DiscoveredDevice& Odin2Device::GetDiscoveredDevice() const {
  return discovered_device_;
}

OdinResult Odin2Device::GetFirmwareVersion(std::string& version, uint32_t timeout_ms) {
  std::vector<uint8_t> payload;

  LOG_DEBUG("GetFirmwareVersion\n");

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kVersionQuery), payload,
                                  response, timeout_ms)) {
    return OdinResult::kTimeout;
  }

  if (response.result != OdinResult::kOk) {
    return response.result;
  }

  // Parse version from payload: [ret_code, major, minor, patch, ...]
  if (response.response.payload.size() >= 4) {
    uint8_t major = response.response.payload[1];
    uint8_t minor = response.response.payload[2];
    uint8_t patch = response.response.payload[3];
    version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  } else {
    version = "unknown";
  }

  return OdinResult::kOk;
}

OdinResult Odin2Device::SetOperatingMode(OdinOperatingMode mode, uint32_t timeout_ms) {
  // SetMode payload format: [0]: target_mode (1 byte) - 0x00=Standby, 0x01=Normal
  std::vector<uint8_t> payload;
  payload.push_back(static_cast<uint8_t>(mode));

  LOG_DEBUG("SetOperatingMode: mode=%d\n", static_cast<int>(mode));

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kSetMode), payload, response,
                                  timeout_ms)) {
    return OdinResult::kTimeout;
  }

  return response.result;
}

OdinResult Odin2Device::SetSensorMode(uint8_t mode, uint32_t timeout_ms) {
  std::vector<uint8_t> payload;
  payload.push_back(0x00);
  payload.push_back(mode);

  LOG_DEBUG("SetSensorModeSync: mode=%d\n", mode);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kSensorMode), payload,
                                  response, timeout_ms)) {
    LOG_ERROR("Failed to send sensor mode: 0x%02x\n", mode);
    return OdinResult::kTimeout;
  }
  return response.result;
}

void Odin2Device::EnableSlamOdomSync(bool enabled, uint32_t max_frame_lag) {
  if (slam_odom_sync_) {
    slam_odom_sync_->SetEnabled(enabled);
    slam_odom_sync_->SetMaxFrameLag(max_frame_lag);
    LOG_INFO("Odin2Device[%d]: SLAM-Odom sync %s (max_lag=%u)\n", handle_,
             enabled ? "enabled" : "disabled", max_frame_lag);
  }
}

OdinResult Odin2Device::StartStream(OdinDataChannel channel, OdinTransportMode transport,
                                    const OdinStreamCfg* mode, uint32_t timeout_ms) {

  // Validate: mode parameter only valid for Image channels
  bool is_image_channel = (channel == OdinDataChannel::kImage0 || 
                           channel == OdinDataChannel::kImage1);
  if (mode != nullptr && !is_image_channel) {
    LOG_ERROR("StartStream: mode parameter only valid for Image channels\n");
    return OdinResult::kInvalidArgument;
  }

  // Get pointer to port and data channel for this stream type
  uint16_t* port_ptr = nullptr;
  std::unique_ptr<ITransport>* channel_ptr = nullptr;
  TransportReceiveCallback callback;

  // Store transport mode for TCP frame reassembly
  current_transport_mode_ = transport;

  switch (channel) {
    case OdinDataChannel::kRawPoint:
      port_ptr = &ports_.raw_point;
      channel_ptr = &point_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpData(channel, transport, data, len);
      };
      break;
    case OdinDataChannel::kSlamPoint:
      port_ptr = &ports_.slam;
      channel_ptr = &slam_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpData(channel, transport, data, len);
      };
      break;
    case OdinDataChannel::kImage0:
      port_ptr = &ports_.jpeg;
      channel_ptr = &jpeg_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpData(channel, transport, data, len);
      };
      break;
    case OdinDataChannel::kImage1:
      port_ptr = &ports_.jpeg2;
      channel_ptr = &jpeg2_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpData(channel, transport, data, len);
      };
      break;
    case OdinDataChannel::kImu:
      port_ptr = &ports_.imu;
      channel_ptr = &imu_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpImuData(channel, transport, data, len);
      };
      break;
    case OdinDataChannel::kOdom:
      port_ptr = &ports_.odom;
      channel_ptr = &odom_channel_;
      callback = [this, channel, transport](const uint8_t* data, size_t len, const ITransportAddress&) {
        OnTcpOrUdpData(channel, transport, data, len);
      };
      break;
    default:
      LOG_ERROR("StartStream: unknown channel 0x%02x\n", static_cast<uint8_t>(channel));
      return OdinResult::kInvalidArgument;
  }

  // Allocate port if not already set
  uint16_t dst_port = *port_ptr;
  if (dst_port == 0) {
    dst_port = GenerateAvailablePort();
    if (dst_port == 0) {
      LOG_ERROR("StartStream: failed to allocate available port for channel 0x%02x\n",
                static_cast<uint8_t>(channel));
      return OdinResult::kUnknownError;
    }
    *port_ptr = dst_port;
  }

  auto channel_name = [](OdinDataChannel ch) -> const char* {
    switch (ch) {
      case OdinDataChannel::kRawPoint:  return "RawPoint";
      case OdinDataChannel::kSlamPoint: return "SlamPoint";
      case OdinDataChannel::kImage0:    return "Image0";
      case OdinDataChannel::kImage1:    return "Image1";
      case OdinDataChannel::kImu:       return "IMU";
      case OdinDataChannel::kOdom:      return "Odom";
      default: return "Unknown";
    }
  };
  const char* transport_name = (transport == OdinTransportMode::kTcp) ? "TCP" : "UDP";

  // Build TLV payload
  std::vector<uint8_t> payload;
  payload.reserve(mode ? 20 : 8);

  // data_type (1 byte)
  payload.push_back(static_cast<uint8_t>(channel));

  // TLV: dst_port
  payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kDstPort));
  payload.push_back(0x02);
  payload.push_back(static_cast<uint8_t>((dst_port >> 8) & 0xFF));
  payload.push_back(static_cast<uint8_t>(dst_port & 0xFF));

  // TLV: transport_mode
  payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kTransportMode));
  payload.push_back(0x01);
  payload.push_back(static_cast<uint8_t>(transport));

  // Add mode TLVs if provided (Image channels only)
  // Look up resolution_id and fps_id from internal capability table
  if (mode && (mode->width > 0 || mode->height > 0)) {
    uint8_t resolution_id = 0xFF;
    uint8_t fps_id = 0xFF;
    
    // Look up indices from capability table
    {
      std::lock_guard<std::mutex> cap_lock(capability_mutex_);
      auto it = capability_table_.find(channel);
      if (it != capability_table_.end()) {
        const auto& configs = it->second;
        for (size_t i = 0; i < configs.size(); ++i) {
          if (configs[i].width == mode->width && configs[i].height == mode->height) {
            resolution_id = static_cast<uint8_t>(i);
            // For fps_id, we assume fps is stored in the config
            // If fps matches, use fps_id = 0 (first fps in this resolution)
            if (configs[i].fps == mode->fps || mode->fps == 0) {
              fps_id = 0;
            }
            break;
          }
        }
      }
    }
    
    LOG_INFO("StartStream: %s mode %ux%u@%u -> resolution_id=%u, fps_id=%u\n",
             channel_name(channel), mode->width, mode->height, mode->fps, resolution_id, fps_id);
    
    // TLV: resolution_id (0xFF = not specified)
    if (resolution_id != 0xFF) {
      payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kResolutionId));
      payload.push_back(0x01);
      payload.push_back(resolution_id);
    }

    // TLV: fps_id (0xFF = not specified)
    if (fps_id != 0xFF) {
      payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kFpsId));
      payload.push_back(0x01);
      payload.push_back(fps_id);
    }

    // TLV: format (kUnknown = not specified)
    if (mode->format != OdinDataFormat::kUnknown) {
      payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kFormat));
      payload.push_back(0x01);
      payload.push_back(static_cast<uint8_t>(mode->format));
      LOG_INFO("StartStream: %s format=0x%02x\n", channel_name(channel), static_cast<uint8_t>(mode->format));
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kChannelConfig), payload,
                                  response, timeout_ms)) {
    LOG_ERROR("StartStream: %s(%u) %s timeout\n", channel_name(channel), dst_port, transport_name);
    return OdinResult::kTimeout;
  }

  if (response.response.payload.empty()) {
    LOG_ERROR("StartStream: %s(%u) %s empty response\n", channel_name(channel), dst_port, transport_name);
    return OdinResult::kUnknownError;
  }

  uint8_t ret_code = response.response.payload[0];
  if (ret_code != 0) {
    LOG_ERROR("StartStream: %s(%u) %s failed, code=%d\n", 
              channel_name(channel), dst_port, transport_name, ret_code);
    return OdinResult::kUnknownError;
  }

   // Start data channel to receive data on this port
  if (*channel_ptr == nullptr) {
    if (transport == OdinTransportMode::kTcp) {
      channel_ptr->reset(new TcpTransport());
    } else {
      channel_ptr->reset(new UdpTransport());
    }

    NetworkAddress addr;
    addr.ip = host_ip_;
    addr.port = dst_port;
    if (!(*channel_ptr)->Open(addr)) {
      LOG_ERROR("StartStream: %s(%u) %s open failed\n", 
                channel_name(channel), dst_port, transport_name);
      channel_ptr->reset();
      return OdinResult::kSocketError;
    }
    (*channel_ptr)->SetReceiveCallback(std::move(callback));
    if (!(*channel_ptr)->StartReceiving()) {
      LOG_ERROR("StartStream: %s(%u) %s listen failed\n", 
                channel_name(channel), dst_port, transport_name);
      (*channel_ptr)->Close();
      channel_ptr->reset();
      return OdinResult::kSocketError;
    }
  }

  LOG_INFO("StartStream: %s(%u) %s ok\n", channel_name(channel), dst_port, transport_name);
  return OdinResult::kOk;
}

OdinResult Odin2Device::GetSensorCapability(std::vector<OdinSensorCapability>& capabilities,
                                            const std::vector<OdinDataChannel>& channels,
                                            uint32_t timeout_ms) {

  capabilities.clear();

  // Build payload: list of channel types to query (empty = query all)
  std::vector<uint8_t> payload;
  for (auto ch : channels) {
    payload.push_back(static_cast<uint8_t>(ch));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kCmdIdQueryCapability), payload,
                                  response, timeout_ms)) {
    LOG_ERROR("GetSensorCapability: failed to send command (timeout)\n");
    return OdinResult::kTimeout;
  }

  const auto& resp_payload = response.response.payload;
  if (resp_payload.empty()) {
    LOG_ERROR("GetSensorCapability: empty response\n");
    return OdinResult::kUnknownError;
  }

  // Check if the payload is in JSON format (starting with '{')
  if (resp_payload[0] == '{') {
    // Parse JSON payload
    std::string json_str(resp_payload.begin(), resp_payload.end());
    cJSON *root = cJSON_Parse(json_str.c_str());
    if (!root) {
      LOG_ERROR("GetSensorCapability: failed to parse JSON payload\n");
      return OdinResult::kUnknownError;
    }

    cJSON *channels_arr = cJSON_GetObjectItem(root, "channels");
    if (channels_arr && cJSON_IsArray(channels_arr)) {
      int arr_size = cJSON_GetArraySize(channels_arr);
      for (int i = 0; i < arr_size; ++i) {
        cJSON *ch_obj = cJSON_GetArrayItem(channels_arr, i);
        if (!ch_obj || !cJSON_IsObject(ch_obj)) continue;

        OdinSensorCapability cap;

        // Parse data_type
        cJSON *data_type = cJSON_GetObjectItem(ch_obj, "data_type");
        if (data_type && cJSON_IsNumber(data_type)) {
          cap.channel = static_cast<OdinDataChannel>(data_type->valueint);
        }

        // Parse format (store in first config if needed)
        cJSON *format = cJSON_GetObjectItem(ch_obj, "format");
        OdinDataFormat channel_format = OdinDataFormat::kUnknown;
        if (format && cJSON_IsNumber(format)) {
          channel_format = static_cast<OdinDataFormat>(format->valueint);
        }

        // Parse resolutions array
        cJSON *resolutions = cJSON_GetObjectItem(ch_obj, "resolutions");
        if (resolutions && cJSON_IsArray(resolutions)) {
          int res_count = cJSON_GetArraySize(resolutions);
          for (int j = 0; j < res_count; ++j) {
            cJSON *res_obj = cJSON_GetArrayItem(resolutions, j);
            if (!res_obj || !cJSON_IsObject(res_obj)) continue;

            OdinStreamCfg cfg;
            cfg.format = channel_format;

            cJSON *width = cJSON_GetObjectItem(res_obj, "width");
            if (width && cJSON_IsNumber(width)) {
              cfg.width = static_cast<uint16_t>(width->valueint);
            }

            cJSON *height = cJSON_GetObjectItem(res_obj, "height");
            if (height && cJSON_IsNumber(height)) {
              cfg.height = static_cast<uint16_t>(height->valueint);
            }

            // fps is an array, create a config for each fps value
            cJSON *fps_arr = cJSON_GetObjectItem(res_obj, "fps");
            if (fps_arr && cJSON_IsArray(fps_arr)) {
              int fps_count = cJSON_GetArraySize(fps_arr);
              for (int k = 0; k < fps_count; ++k) {
                cJSON *fps_val = cJSON_GetArrayItem(fps_arr, k);
                if (fps_val && cJSON_IsNumber(fps_val)) {
                  OdinStreamCfg cfg_copy = cfg;
                  cfg_copy.fps = static_cast<uint16_t>(fps_val->valueint);
                  cap.modes.push_back(cfg_copy);
                }
              }
            } else {
              // No fps array, just add the config with fps=0
              cap.modes.push_back(cfg);
            }
          }
        }

        // Filter by channels if specified
        if (channels.empty()) {
          capabilities.push_back(cap);
          LOG_INFO("GetSensorCapability: channel=0x%02x, %zu configs\n",
                   static_cast<uint8_t>(cap.channel), cap.modes.size());
        } else {
          for (auto ch : channels) {
            if (ch == cap.channel) {
              capabilities.push_back(cap);
              LOG_INFO("GetSensorCapability: channel=0x%02x, %zu configs\n",
                       static_cast<uint8_t>(cap.channel), cap.modes.size());
              break;
            }
          }
        }
      }
    }

    cJSON_Delete(root);
    
    // Update internal capability table
    {
      std::lock_guard<std::mutex> cap_lock(capability_mutex_);
      for (const auto& cap : capabilities) {
        capability_table_[cap.channel] = cap.modes;
      }
    }
    
    LOG_INFO("GetSensorCapability success (JSON): %zu channels (filtered)\n", capabilities.size());
    return OdinResult::kOk;
  }

  // Binary TLV format fallback
  if (resp_payload.size() < 2) {
    LOG_ERROR("GetSensorCapability: response too short (%zu bytes)\n", resp_payload.size());
    return OdinResult::kUnknownError;
  }

  uint8_t ret_code = resp_payload[0];
  if (ret_code != 0) {
    LOG_ERROR("GetSensorCapability: device returned error code %d\n", ret_code);
    return OdinResult::kUnknownError;
  }

  uint8_t channel_count = resp_payload[1];
  size_t offset = 2;

  // Parse binary TLV response
  for (uint8_t i = 0; i < channel_count && offset < resp_payload.size(); i++) {
    if (offset + 2 > resp_payload.size()) break;

    OdinSensorCapability cap;
    cap.channel = static_cast<OdinDataChannel>(resp_payload[offset++]);
    uint8_t mode_count = resp_payload[offset++];

    // Each config is 6 bytes: [width_h, width_l, height_h, height_l, fps, format]
    for (uint8_t j = 0; j < mode_count && offset + 6 <= resp_payload.size(); j++) {
      OdinStreamCfg cfg;
      cfg.width = (static_cast<uint16_t>(resp_payload[offset]) << 8) |
                 static_cast<uint16_t>(resp_payload[offset + 1]);
      cfg.height = (static_cast<uint16_t>(resp_payload[offset + 2]) << 8) |
                  static_cast<uint16_t>(resp_payload[offset + 3]);
      cfg.fps = resp_payload[offset + 4];
      cfg.format = static_cast<OdinDataFormat>(resp_payload[offset + 5]);
      offset += 6;
      cap.modes.push_back(cfg);
    }

    // Filter by channels if specified
    if (channels.empty()) {
      capabilities.push_back(cap);
      LOG_INFO("GetSensorCapability: channel=0x%02x, %zu configs\n",
               static_cast<uint8_t>(cap.channel), cap.modes.size());
    } else {
      for (auto ch : channels) {
        if (ch == cap.channel) {
          capabilities.push_back(cap);
          LOG_INFO("GetSensorCapability: channel=0x%02x, %zu configs\n",
                   static_cast<uint8_t>(cap.channel), cap.modes.size());
          break;
        }
      }
    }
  }

  // Update internal capability table
  {
    std::lock_guard<std::mutex> cap_lock(capability_mutex_);
    for (const auto& cap : capabilities) {
      capability_table_[cap.channel] = cap.modes;
    }
  }

  LOG_INFO("GetSensorCapability success (binary): %zu channels (filtered)\n", capabilities.size());
  return OdinResult::kOk;
}

OdinResult Odin2Device::CloseStream(OdinDataChannel channel, uint32_t timeout_ms) {
  // Build TLV payload with dst_port=0 to close stream
  std::vector<uint8_t> payload;
  payload.reserve(5);

  // data_type (1 byte)
  payload.push_back(static_cast<uint8_t>(channel));

  // TLV: dst_port = 0 (unsubscribe)
  payload.push_back(static_cast<uint8_t>(ChannelConfigTlvType::kDstPort));
  payload.push_back(0x02);  // length
  payload.push_back(0x00);  // port high byte = 0
  payload.push_back(0x00);  // port low byte = 0

  LOG_DEBUG("CloseStream: channel=0x%02x\n", static_cast<uint8_t>(channel));

  std::lock_guard<std::mutex> lock(mutex_);
  if (!command_channel_) return OdinResult::kNotInitialized;

  OdinCommandSyncResponse response;
  if (!command_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kChannelConfig), payload,
                                  response, timeout_ms)) {
    LOG_ERROR("CloseStream: failed to send command (timeout)\n");
    return OdinResult::kTimeout;
  }

  // Check response ret_code (first byte of payload)
  if (response.response.payload.empty()) {
    LOG_ERROR("CloseStream: empty response payload\n");
    return OdinResult::kUnknownError;
  }

  uint8_t ret_code = response.response.payload[0];
  if (ret_code != 0) {
    LOG_ERROR("CloseStream: device returned error code %d\n", ret_code);
    return OdinResult::kUnknownError;
  }

  LOG_INFO("CloseStream success: channel=0x%02x\n", static_cast<uint8_t>(channel));
  return OdinResult::kOk;
}

void Odin2Device::RegisterPointCloudCallback(OdinPointCloudCallback cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  point_callback_ = cb;
  point_user_data_ = user_data;
}

void Odin2Device::RegisterSlamCallback(OdinSlamCallback cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  slam_callback_ = cb;
  slam_user_data_ = user_data;
  // Update synchronizer callbacks
  if (slam_odom_sync_) {
    slam_odom_sync_->SetCallbacks(slam_callback_, slam_user_data_, odom_callback_, odom_user_data_);
  }
}

void Odin2Device::RegisterImageCallback(OdinImageCallback cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  image_callback_ = cb;
  image_user_data_ = user_data;
}

void Odin2Device::RegisterImageCallback2(OdinImageCallback2 cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  image_callback_2_ = cb;
  image_user_data_2_ = user_data;
}

void Odin2Device::RegisterImuCallback(OdinImuCallback cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  imu_callback_ = cb;
  imu_user_data_ = user_data;
}

void Odin2Device::RegisterOdomCallback(OdinOdomCallback cb, void* user_data) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  odom_callback_ = cb;
  odom_user_data_ = user_data;
  // Update synchronizer callbacks
  if (slam_odom_sync_) {
    slam_odom_sync_->SetCallbacks(slam_callback_, slam_user_data_, odom_callback_, odom_user_data_);
  }
}

void Odin2Device::StopDataChannels() {
  // Helper lambda to stop an ITransport channel (UDP or TCP)
  auto stopChannel = [](std::unique_ptr<ITransport>& channel) {
    if (channel) {
      channel->StopReceiving();
      channel->Close();
      channel.reset();
    }
  };

  stopChannel(point_channel_);
  stopChannel(slam_channel_);
  stopChannel(jpeg_channel_);
  stopChannel(jpeg2_channel_);
  stopChannel(imu_channel_);
  stopChannel(odom_channel_);
}

uint16_t Odin2Device::ExpectedPointUdpCount(OdinDataChannel channel) const {
  // Match device-side packing: integer-truncated points per packet, then ceil
  constexpr size_t kTotalPoints = kOdin2PclWidth * kOdin2PclHeight;
  constexpr size_t kRawPtsPerPkt = kOdin2UdpMaxDataSize / sizeof(OdinRawPoint<uint16_t>);
  constexpr size_t kSlamPtsPerPkt = kOdin2UdpMaxDataSize / sizeof(OdinSlamPoint<uint16_t>);
  constexpr uint16_t kRawPointFramePackets =
      static_cast<uint16_t>((kTotalPoints + kRawPtsPerPkt - 1) / kRawPtsPerPkt);
  constexpr uint16_t kSlamPointFramePackets =
      static_cast<uint16_t>((kTotalPoints + kSlamPtsPerPkt - 1) / kSlamPtsPerPkt);

  if (channel == OdinDataChannel::kRawPoint) return kRawPointFramePackets;
  if (channel == OdinDataChannel::kSlamPoint) return kSlamPointFramePackets;
  return 0;
}

// =============================================================================
// Packet Parsing
// =============================================================================

bool Odin2Device::ParsePointCloudPacket(const uint8_t* data, size_t length,
                                        OdinPointCloudPacket* packet) {
  if (!data || !packet || length < kOdin2DataHeaderSize) return false;

  const OdinDataFrameHeader* header = reinterpret_cast<const OdinDataFrameHeader*>(data);
  uint16_t frame_length = header->length;
  
  if (frame_length > length || frame_length < kOdin2DataHeaderSize) return false;

  size_t payload_size = frame_length - kOdin2DataHeaderSize;
  const uint8_t* payload_ptr = data + kOdin2DataHeaderSize;

  packet->version = header->version;
  packet->length = frame_length;
  packet->point_count = header->dot_or_sample_count;
  packet->udp_count = header->udp_count;
  packet->frame_count = header->frame_count;
  packet->data_type = header->data_type;
  packet->time_type = header->time_type;
  packet->reserved = header->reserved;
  packet->timestamp = header->timestamp;
  packet->payload.assign(payload_ptr, payload_ptr + payload_size);
  return true;
}

bool Odin2Device::ParseImagePacket(const uint8_t* data, size_t length, OdinImagePacket* packet) {
  if (!data || !packet || length < kOdin2DataHeaderSize) return false;

  const OdinDataFrameHeader* header = reinterpret_cast<const OdinDataFrameHeader*>(data);
  uint16_t frame_length = header->length;
  if (frame_length > length || frame_length < kOdin2DataHeaderSize) return false;

  size_t payload_size = frame_length - kOdin2DataHeaderSize;
  const uint8_t* payload_ptr = data + kOdin2DataHeaderSize;

  packet->version = header->version;
  packet->length = frame_length;
  packet->udp_count = header->udp_count;
  packet->frame_count = header->frame_count;
  packet->timestamp = header->timestamp;
  packet->payload.assign(payload_ptr, payload_ptr + payload_size);
  return true;
}

bool Odin2Device::ParseImuPacket(const uint8_t* data, size_t length, OdinImuPacket* packet) {
  if (!data || !packet || length < kOdin2DataHeaderSize) return false;

  const OdinDataFrameHeader* header = reinterpret_cast<const OdinDataFrameHeader*>(data);
  uint16_t frame_length = header->length;
  if (frame_length > length || frame_length < kOdin2DataHeaderSize) return false;

  size_t payload_size = frame_length - kOdin2DataHeaderSize;
  const uint8_t* payload_ptr = data + kOdin2DataHeaderSize;

  packet->version = header->version;
  packet->length = frame_length;
  packet->sample_count = header->dot_or_sample_count;
  packet->timestamp = header->timestamp;
  packet->payload.assign(payload_ptr, payload_ptr + payload_size);

  return true;
}

bool Odin2Device::ParseOdomPacket(const uint8_t* data, size_t length, OdinOdomPacket* packet) {
  if (!data || !packet || length < kOdin2DataHeaderSize) return false;

  const OdinDataFrameHeader* header = reinterpret_cast<const OdinDataFrameHeader*>(data);
  uint16_t frame_length = header->length;
  if (frame_length > length || frame_length < kOdin2DataHeaderSize) return false;

  size_t payload_size = frame_length - kOdin2DataHeaderSize;
  const uint8_t* payload_ptr = data + kOdin2DataHeaderSize;

  packet->version = header->version;
  packet->length = frame_length;
  packet->data_count = header->dot_or_sample_count;
  packet->frame_count = header->frame_count;
  packet->timestamp = header->timestamp;
  packet->payload.assign(payload_ptr, payload_ptr + payload_size);

  return true;
}

// =============================================================================
// Frame Processing
// =============================================================================

void Odin2Device::FillLostPacketsWithZeros(PointFrameState& state, uint16_t lost_packets,
                                           size_t points_per_packet) {
  if (lost_packets == 0 || points_per_packet == 0) return;
  size_t lost_points = static_cast<size_t>(lost_packets) * points_per_packet;
  size_t fill_bytes = lost_points * sizeof(OdinRawPoint<uint16_t>);

  LOG_WARN("Point cloud packet loss: lost %u packets (%zu points)\n", lost_packets, lost_points);

  std::vector<uint8_t> zero_fill(fill_bytes, 0);
  state.packet.payload.insert(state.packet.payload.end(), zero_fill.begin(), zero_fill.end());

  uint32_t new_total = static_cast<uint32_t>(state.packet.point_count) + lost_points;
  if (new_total > 65535) new_total = 65535;
  state.packet.point_count = static_cast<uint16_t>(new_total);
  state.udp_segments += lost_packets;
}

bool Odin2Device::DetectJpegTerminator(ImageFrameState& state, const std::vector<uint8_t>& chunk) {
  if (chunk.empty()) return false;

  if (state.last_byte_valid && state.last_byte == 0xFF && chunk[0] == 0xD9) return true;

  for (size_t i = 1; i < chunk.size(); ++i) {
    if (chunk[i - 1] == 0xFF && chunk[i] == 0xD9) return true;
  }

  state.last_byte = chunk.back();
  state.last_byte_valid = true;
  return false;
}

void Odin2Device::EmitPointFrame(PointFrameState& state) {
  if (!state.active || state.packet.payload.empty()) {
    state.Reset();
    return;
  }

  state.packet.udp_count = state.udp_segments;
  size_t total_length = kOdin2DataHeaderSize + state.packet.payload.size();
  if (total_length > 65535) total_length = 65535;
  state.packet.length = static_cast<uint16_t>(total_length);
  state.packet.device = handle_;

  OdinDataChannel channel = static_cast<OdinDataChannel>(state.packet.data_type);

  // Get callbacks
  OdinPointCloudCallback point_cb = nullptr;
  void* point_user = nullptr;
  OdinSlamCallback slam_cb = nullptr;
  void* slam_user = nullptr;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    point_cb = point_callback_;
    point_user = point_user_data_;
    slam_cb = slam_callback_;
    slam_user = slam_user_data_;
  }

  if (channel == OdinDataChannel::kRawPoint && point_cb) {
    // Convert uint16_t to float
    size_t src_point_size = sizeof(OdinRawPoint<uint16_t>);
    size_t point_count = state.packet.payload.size() / src_point_size;
    const OdinRawPoint<uint16_t>* src =
        reinterpret_cast<const OdinRawPoint<uint16_t>*>(state.packet.payload.data());

    std::vector<uint8_t> new_payload(point_count * sizeof(OdinRawPoint<float>));
    OdinRawPoint<float>* dst = reinterpret_cast<OdinRawPoint<float>*>(new_payload.data());

    for (size_t i = 0; i < point_count; ++i) {
      dst[i].x = static_cast<float>(src[i].x);
      dst[i].y = static_cast<float>(src[i].y) - 30000.0f;
      dst[i].z = static_cast<float>(src[i].z) - 30000.0f;
      dst[i].intensity = src[i].intensity;
      dst[i].confidence = src[i].confidence;
    }
    state.packet.payload = std::move(new_payload);
    point_cb(state.packet, point_user);
  } else if (channel == OdinDataChannel::kSlamPoint) {
    // Check if sync is enabled - if so, send through synchronizer which will do conversion
    bool sync_enabled = slam_odom_sync_ && slam_odom_sync_->IsEnabled();

    if (sync_enabled) {
      // Send uint16_t format to synchronizer for transformation
      // The synchronizer will convert to float and call the user callback
      slam_odom_sync_->ProcessSlam(state.packet);
    } else {
      // No sync - convert SLAM points to float format directly
      if (slam_cb) {
        size_t src_point_size = sizeof(OdinSlamPoint<uint16_t>);
        size_t point_count = state.packet.payload.size() / src_point_size;
        const OdinSlamPoint<uint16_t>* src =
            reinterpret_cast<const OdinSlamPoint<uint16_t>*>(state.packet.payload.data());

        std::vector<uint8_t> new_payload(point_count * sizeof(OdinSlamPoint<float>));
        OdinSlamPoint<float>* dst = reinterpret_cast<OdinSlamPoint<float>*>(new_payload.data());

        for (size_t i = 0; i < point_count; ++i) {
          dst[i].x = static_cast<float>(src[i].x);
          dst[i].y = static_cast<float>(src[i].y) - 30000.0f;
          dst[i].z = static_cast<float>(src[i].z) - 30000.0f;
          dst[i].r = src[i].r;
          dst[i].g = src[i].g;
          dst[i].b = src[i].b;
          dst[i].a = src[i].a;
        }
        state.packet.payload = std::move(new_payload);
        slam_cb(state.packet, slam_user);
      }
    }
  }

  state.Reset();
}

void Odin2Device::EmitImageFrame(ImageFrameState& state) {
  if (!state.active || state.packet.payload.empty()) {
    state.Reset();
    return;
  }

  // Firmware format: [JPEG data without EOI] + [4-byte expected length] + [EOI 0xFFD9]
  // expected_length = JPEG data + EOI = complete JPEG length
  // Validate: (payload - 4) must equal expected_length
  constexpr size_t kLengthFieldSize = 4;
  constexpr size_t kEOISize = 2;  // 0xFFD9

  if (state.packet.payload.size() <
      kLengthFieldSize + kEOISize + 2) {  // minimum: SOI(2) + length(4) + EOI(2)
    state.Reset();
    return;
  }

  // Length field is 4 bytes before the EOI terminator
  size_t payload_size = state.packet.payload.size();
  const uint8_t* len_ptr = state.packet.payload.data() + payload_size - kLengthFieldSize - kEOISize;
  uint32_t expected_len =
      static_cast<uint32_t>(len_ptr[0]) << 24 | (static_cast<uint32_t>(len_ptr[1]) << 16) |
      (static_cast<uint32_t>(len_ptr[2]) << 8) | (static_cast<uint32_t>(len_ptr[3]));

  // Actual JPEG length = payload - 4 (length field) = JPEG data + EOI
  uint32_t actual_len = static_cast<uint32_t>(payload_size - kLengthFieldSize);

  if (actual_len != expected_len) {
    // Data lost or corrupted, discard this frame
    LOG_WARN("Image frame dropped: expected %u bytes, got %u (lost %d)\n", expected_len, actual_len,
             static_cast<int>(expected_len) - static_cast<int>(actual_len));
    state.Reset();
    return;
  }

  // Remove length field, reconstruct complete JPEG: [JPEG data] + [EOI]
  // Move EOI to where length field was
  uint8_t* data = state.packet.payload.data();
  size_t jpeg_data_len = payload_size - kLengthFieldSize - kEOISize;
  data[jpeg_data_len] = 0xFF;
  data[jpeg_data_len + 1] = 0xD9;
  state.packet.payload.resize(actual_len);

  state.packet.udp_count = state.udp_segments;
  size_t total_length = kOdin2DataHeaderSize + state.packet.payload.size();
  if (total_length > 65535) total_length = 65535;
  state.packet.length = static_cast<uint16_t>(total_length);
  state.packet.device = handle_;

  // Determine which callback to use based on which state this is
  OdinImageCallback image_cb = nullptr;
  void* image_user = nullptr;
  OdinImageCallback2 image_cb2 = nullptr;
  void* image_user2 = nullptr;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    image_cb = image_callback_;
    image_user = image_user_data_;
    image_cb2 = image_callback_2_;
    image_user2 = image_user_data_2_;
  }

  if (&state == &image_state_ && image_cb) {
    image_cb(state.packet, image_user);
  } else if (&state == &image2_state_ && image_cb2) {
    image_cb2(state.packet, image_user2);
  }

  state.Reset();
}

void Odin2Device::ProcessPointFragment(OdinDataChannel channel, const OdinPointCloudPacket& fragment) {
  if (channel != OdinDataChannel::kRawPoint && channel != OdinDataChannel::kSlamPoint) return;

  uint16_t expected_packets = ExpectedPointUdpCount(channel);
  std::lock_guard<std::mutex> lock(frame_mutex_);
  PointFrameState& state =
      (channel == OdinDataChannel::kSlamPoint) ? slam_point_state_ : raw_point_state_;

  bool start_new_frame =
      state.active && (fragment.frame_count != state.packet.frame_count ||
                       (state.udp_segments > 0 && fragment.udp_count <= state.last_udp_seq) ||
                       (fragment.udp_count == 0 && state.udp_segments > 0));

  if (start_new_frame) {
    if (channel == OdinDataChannel::kRawPoint && expected_packets > 0 &&
        state.udp_segments < expected_packets && state.udp_segments > 0 &&
        !state.packet.payload.empty()) {
      uint16_t tail_lost = expected_packets - state.udp_segments;
      size_t points_per_pkt =
          state.packet.payload.size() / state.udp_segments / sizeof(OdinRawPoint<uint16_t>);
      FillLostPacketsWithZeros(state, tail_lost, points_per_pkt);
    }
    EmitPointFrame(state);
  }

  if (!state.active) {
    state.active = true;
    state.packet = OdinPointCloudPacket();
    state.packet.device = handle_;
    state.packet.version = fragment.version;
    state.packet.frame_count = fragment.frame_count;
    state.packet.data_type = fragment.data_type;
    state.packet.time_type = fragment.time_type;
    state.packet.reserved = fragment.reserved;
    state.packet.timestamp = fragment.timestamp;
    state.packet.point_count = 0;
    state.udp_segments = 0;
    state.last_udp_seq = 0;
    if (expected_packets > 0 && !fragment.payload.empty()) {
      state.packet.payload.reserve(fragment.payload.size() * expected_packets);
    }
  }

  if (!state.active) return;

  // Handle packet loss
  if (channel == OdinDataChannel::kRawPoint && state.udp_segments > 0 &&
      fragment.udp_count > state.last_udp_seq + 1 && !fragment.payload.empty()) {
    uint16_t lost = fragment.udp_count - state.last_udp_seq - 1;
    size_t points_per_pkt = fragment.payload.size() / sizeof(OdinRawPoint<uint16_t>);
    FillLostPacketsWithZeros(state, lost, points_per_pkt);
  }

  if (!fragment.payload.empty()) {
    state.packet.payload.insert(state.packet.payload.end(), fragment.payload.begin(),
                                fragment.payload.end());
  }

  uint32_t total_points = static_cast<uint32_t>(state.packet.point_count) + fragment.point_count;
  if (total_points > 65535) total_points = 65535;
  state.packet.point_count = static_cast<uint16_t>(total_points);
  ++state.udp_segments;
  state.last_udp_seq = fragment.udp_count;

  if (expected_packets > 0 && state.udp_segments >= expected_packets) {
    EmitPointFrame(state);
  }
}

void Odin2Device::ProcessImageFragment(OdinDataChannel channel, const OdinImagePacket& fragment) {
  if (channel != OdinDataChannel::kImage0 && channel != OdinDataChannel::kImage1) return;

  std::lock_guard<std::mutex> lock(frame_mutex_);
  ImageFrameState& state = (channel == OdinDataChannel::kImage1) ? image2_state_ : image_state_;

  bool start_new_frame =
      state.active && (fragment.frame_count != state.packet.frame_count ||
                       (state.udp_segments > 0 && fragment.udp_count <= state.last_udp_seq) ||
                       (fragment.udp_count == 0 && state.udp_segments > 0));

  if (start_new_frame) {
    EmitImageFrame(state);
  }

  if (!state.active) {
    state.active = true;
    state.packet = OdinImagePacket();
    state.packet.device = handle_;
    state.packet.version = fragment.version;
    state.packet.frame_count = fragment.frame_count;
    state.packet.timestamp = fragment.timestamp;
    state.udp_segments = 0;
    state.last_udp_seq = 0;
    state.last_byte = 0;
    state.last_byte_valid = false;
    if (!fragment.payload.empty()) {
      state.packet.payload.reserve(fragment.payload.size() * 4);
    }
  }

  if (!state.active) return;

  if (!fragment.payload.empty()) {
    state.packet.payload.insert(state.packet.payload.end(), fragment.payload.begin(),
                                fragment.payload.end());
  }

  ++state.udp_segments;
  state.last_udp_seq = fragment.udp_count;

  if (DetectJpegTerminator(state, fragment.payload)) {
    EmitImageFrame(state);
  }
}

// =============================================================================
// Data Channel Handlers
// =============================================================================

void Odin2Device::OnDataReceived(const uint8_t* data, size_t length) {
  if (!data || length < kOdin2DataHeaderSize) return;

  const auto* header = reinterpret_cast<const OdinDataFrameHeader*>(data);
  OdinDataChannel channel = static_cast<OdinDataChannel>(header->data_type);

  switch (channel) {
    case OdinDataChannel::kRawPoint:
    case OdinDataChannel::kSlamPoint: {
      OdinPointCloudPacket packet;
      if (ParsePointCloudPacket(data, length, &packet)) {
        ProcessPointFragment(channel, packet);
      }
      break;
    }
    case OdinDataChannel::kImage0:
    case OdinDataChannel::kImage1: {
      OdinImagePacket packet;
      if (ParseImagePacket(data, length, &packet)) {
        ProcessImageFragment(channel, packet);
      }
      break;
    }
    case OdinDataChannel::kOdom: {
      OdinOdomPacket packet;
      if (ParseOdomPacket(data, length, &packet)) {
        packet.device = handle_;

        // Extract odom_type from payload
        OdomSourceType odom_type = OdomSourceType::kLowFrequency;
        if (packet.payload.size() >= sizeof(OdinOdomData)) {
          const OdinOdomData* odom_data =
              reinterpret_cast<const OdinOdomData*>(packet.payload.data());
          odom_type = static_cast<OdomSourceType>(odom_data->type);
        }

        bool is_low_frequency = (odom_type == OdomSourceType::kLowFrequency);
        bool sync_enabled = slam_odom_sync_ && slam_odom_sync_->IsEnabled();

        // Only low-frequency odom participates in SLAM sync
        if (is_low_frequency && sync_enabled) {
          slam_odom_sync_->ProcessOdom(packet, odom_type);
        } else {
          // High-frequency or sync disabled - call user callback directly
          OdinOdomCallback cb = nullptr;
          void* user = nullptr;
          {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            cb = odom_callback_;
            user = odom_user_data_;
          }
          if (cb) cb(packet, odom_type, user);
        }
      }
      break;
    }
    default:
      break;
  }
}

void Odin2Device::OnImuData(const uint8_t* data, size_t length) {
  OdinImuPacket packet;
  if (!ParseImuPacket(data, length, &packet)) return;

  packet.device = handle_;
  OdinImuCallback cb = nullptr;
  void* user = nullptr;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    cb = imu_callback_;
    user = imu_user_data_;
  }
  if (cb) cb(packet, user);
}

// =============================================================================
// TCP/UDP Data Handlers with Frame Reassembly
// =============================================================================

void Odin2Device::OnTcpOrUdpData(OdinDataChannel channel, OdinTransportMode transport,
                                  const uint8_t* data, size_t length) {
  if (transport == OdinTransportMode::kUdp) {
    // UDP: each recv is a complete frame
    OnDataReceived(data, length);
  } else {
    // TCP: need frame reassembly
    {
      std::lock_guard<std::mutex> lock(tcp_buffer_mutex_);
      tcp_buffers_[channel].Append(data, length);
    }
    ProcessTcpBuffer(channel, [this](const uint8_t* frame, size_t len) {
      OnDataReceived(frame, len);
    });
  }
}

void Odin2Device::OnTcpOrUdpImuData(OdinDataChannel channel, OdinTransportMode transport,
                                     const uint8_t* data, size_t length) {
  if (transport == OdinTransportMode::kUdp) {
    // UDP: each recv is a complete frame
    OnImuData(data, length);
  } else {
    // TCP: need frame reassembly
    {
      std::lock_guard<std::mutex> lock(tcp_buffer_mutex_);
      tcp_buffers_[channel].Append(data, length);
    }
    ProcessTcpBuffer(channel, [this](const uint8_t* frame, size_t len) {
      OnImuData(frame, len);
    });
  }
}

void Odin2Device::ProcessTcpBuffer(OdinDataChannel channel,
                                    std::function<void(const uint8_t*, size_t)> handler) {
  constexpr size_t kMinHeaderSize = 3;  // version(1) + length(2)
  
  std::lock_guard<std::mutex> lock(tcp_buffer_mutex_);
  TcpStreamBuffer& buf = tcp_buffers_[channel];
  
  while (buf.Size() >= kMinHeaderSize) {
    // Frame format: [version:1][length:2 little-endian][...]
    uint16_t frame_len = buf.Data()[1] | (static_cast<uint16_t>(buf.Data()[2]) << 8);
    
    // Sanity check
    if (frame_len < kMinHeaderSize || frame_len >= 65535) {
      LOG_WARN("ProcessTcpBuffer: invalid frame length %u, clearing buffer\n", frame_len);
      buf.Clear();
      break;
    }
    
    if (buf.Size() >= frame_len) {
      // Complete frame available
      handler(buf.Data(), frame_len);
      buf.Consume(frame_len);
    } else {
      // Incomplete frame, wait for more data
      break;
    }
  }
}

void Odin2Device::OnFileData(const uint8_t* data, size_t length) {
  std::lock_guard<std::mutex> lock(file_mutex_);
  if (!file_transfer_active_.load()) {
    return;
  }

  if (save_file_fd_ == nullptr) {
    LOG_ERROR("File descriptor is null during file transfer\n");
    return;
  }

  fwrite(data, 1, length, save_file_fd_);
  bytes_transferred_ += length;

  if (bytes_transferred_ >= file_info_.filesize) {
    fclose(save_file_fd_);
    save_file_fd_ = nullptr;
    file_transfer_active_.store(false);
    LOG_INFO("File transfer completed, total bytes: %lu\n",
             static_cast<unsigned long>(bytes_transferred_));
  }
}

namespace {
bool GetFileInfoFromPath(FileInfo& file_info, const std::string& file_path) {
  memset(&file_info, 0, sizeof(FileInfo));
  strncpy(file_info.filename, file_path.c_str(), sizeof(file_info.filename) - 1);

  FILE* fp = std::fopen(file_path.c_str(), "rb");
  if (!fp) {
    LOG_ERROR("Failed to get file info for path: %s\n", file_path.c_str());
    return false;
  }
  std::fseek(fp, 0, SEEK_END);
  file_info.filesize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::fclose(fp);
  return true;
}
}  // namespace

OdinResult Odin2Device::SendFile(OdinFileType type, const std::string& file_path,
                                 OdinUpgradeProgressCallback cb) {
  if (!IsConnected()) {
    return OdinResult::kNotInitialized;
  }

  // For firmware upgrade, use complete OTA flow with HTTP (fallback to HTTPS)
  if (type == OdinFileType::kFirmware) {
    return SendFileFirmwareOta(file_path, cb);
  }

  // For other file types, use simple upload
  // Initialize file transfer: probe HTTP (8080), fall back to HTTPS
  if (!https_transfer_) {
    https_transfer_ = std::make_unique<HttpsFileTransfer>();
    http::Client probe("http://" + device_ip_ + ":8080");
    if (probe.health_check(2000)) {
      https_transfer_->Connect("http://" + device_ip_ + ":8080");
    } else {
      std::string https_url = device_ip_ + ":" + std::to_string(ports_.file);
      if (!https_transfer_->Connect(https_url)) {
        LOG_ERROR("Failed to connect to HTTP server at %s\n", https_url.c_str());
        https_transfer_.reset();
        return OdinResult::kCommunicationError;
      }
    }
  }

  // Map OdinFileType to HTTP resource name for upload
  std::string remote_name;
  switch (type) {
    case OdinFileType::kCalibrationFile:
      remote_name = "calibration";
      break;
    case OdinFileType::kRelocationMap:
      remote_name = "relocation_map";
      break;
    default:
      LOG_ERROR("Unsupported file type for upload: %d\n", static_cast<int>(type));
      return OdinResult::kInvalidArgument;
  }

  LOG_DEBUG("SendFile: type=%s, path=%s\n", remote_name.c_str(), file_path.c_str());

  // Progress callback wrapper
  FileTransferProgressCallback progress_cb = nullptr;
  if (cb) {
    OdinDeviceHandle h = handle_;
    progress_cb = [cb, h](size_t current, size_t total) {
      if (total > 0) {
        float progress = static_cast<float>(current) * 100.0f / static_cast<float>(total);
        cb(h, progress, nullptr);
      }
    };
  }

  auto result = https_transfer_->Upload(file_path, remote_name, progress_cb);
  if (!result.success) {
    LOG_ERROR("Upload failed: %s\n", result.error_msg.c_str());
    return OdinResult::kCommunicationError;
  }

  return OdinResult::kOk;
}

OdinResult Odin2Device::GetFile(OdinFileType type, const std::string& save_path,
                                OdinUpgradeProgressCallback cb) {
  if (!IsConnected()) {
    return OdinResult::kNotInitialized;
  }

  // Initialize file transfer: probe HTTP (8080), fall back to HTTPS
  if (!https_transfer_) {
    https_transfer_ = std::make_unique<HttpsFileTransfer>();
    http::Client probe("http://" + device_ip_ + ":8080");
    if (probe.health_check(2000)) {
      https_transfer_->Connect("http://" + device_ip_ + ":8080");
    } else {
      std::string https_url = device_ip_ + ":" + std::to_string(ports_.file);
      if (!https_transfer_->Connect(https_url)) {
        LOG_ERROR("Failed to connect to HTTP server at %s\n", https_url.c_str());
        https_transfer_.reset();
        return OdinResult::kCommunicationError;
      }
    }
  }

  // Map OdinFileType to HTTP resource name
  std::string remote_name;
  switch (type) {
    case OdinFileType::kRelocationMap:
      remote_name = "relocation_map";
      break;
    case OdinFileType::kCalibrationFile:
      remote_name = "calibration";
      break;
    case OdinFileType::kDevLogFile:
      remote_name = "logs";
      break;
    default:
      LOG_ERROR("Unknown file type: %d\n", static_cast<int>(type));
      return OdinResult::kInvalidArgument;
  }

  LOG_DEBUG("GetFile: type=%s, save_path=%s\n", remote_name.c_str(), save_path.c_str());

  // Progress callback wrapper
  FileTransferProgressCallback progress_cb = nullptr;
  if (cb) {
    OdinDeviceHandle h = handle_;
    progress_cb = [cb, h](size_t current, size_t total) {
      if (total > 0) {
        float progress = static_cast<float>(current) * 100.0f / static_cast<float>(total);
        cb(h, progress, nullptr);
      }
    };
  }

  auto result = https_transfer_->Download(remote_name, save_path, progress_cb);
  if (!result.success) {
    LOG_ERROR("Download failed: %s\n", result.error_msg.c_str());
    return OdinResult::kCommunicationError;
  }

  if (cb) {
    cb(handle_, 100.0, nullptr);
  }
  return OdinResult::kOk;
}

// =============================================================================
// Firmware Upgrade (Simplified OTA Interface)
// =============================================================================

namespace {

// Map OTA state to progress percentage
float MapOtaStateToProgress(const std::string& state, int device_progress) {
  // Progress breakdown:
  // 0-50%: Upload (handled separately)
  // 50-60%: VERIFYING
  // 60-70%: INSTALLING_MCU
  // 70-80%: INSTALLING_SOC
  // 80-95%: REBOOTING
  // 95-100%: POST_VERIFY
  // 100%: DONE
  
  if (state == "UPLOADING") {
    // Upload progress is 0-50%, device_progress is 0-100
    return 50.0f * device_progress / 100.0f;
  } else if (state == "VERIFYING") {
    return 50.0f + 10.0f * device_progress / 100.0f;
  } else if (state == "INSTALLING_MCU") {
    return 60.0f + 10.0f * device_progress / 100.0f;
  } else if (state == "INSTALLING_SOC") {
    return 70.0f + 10.0f * device_progress / 100.0f;
  } else if (state == "REBOOTING") {
    return 80.0f + 15.0f * device_progress / 100.0f;
  } else if (state == "POST_VERIFY") {
    return 95.0f + 5.0f * device_progress / 100.0f;
  } else if (state == "DONE") {
    return 100.0f;
  } else if (state == "FAILED") {
    return -1.0f;  // Indicate failure
  }
  return 0.0f;
}

}  // namespace

OdinResult Odin2Device::SendFileFirmwareOta(const std::string& firmware_path,
                                             OdinUpgradeProgressCallback cb) {
  constexpr int kDefaultHttpPort = 8080;
  constexpr int kTimeoutS = 300;
  
  // 0. Stop heartbeat during OTA - device will reboot and can't respond to heartbeat
  LOG_INFO("SendFileFirmwareOta: Stopping heartbeat for OTA\n");
  StopHeartbeat();
  
  // 1. Set device to upgrade mode
  OdinResult mode_result = SetOperatingMode(OdinOperatingMode::kUpgrade, 3000);
  if (mode_result != OdinResult::kOk) {
    LOG_ERROR("SendFileFirmwareOta: Failed to set upgrade mode\n");
    StartHeartbeat();  // Restart heartbeat on failure
    return mode_result;
  }
  
  // 2. Determine HTTP port - try default 8080 first, then fall back to ports_.file
  int http_port = kDefaultHttpPort;
  {
    std::string probe_url = "http://" + device_ip_ + ":" + std::to_string(kDefaultHttpPort);
    http::Client probe(probe_url);
    if (!probe.health_check(2000) && ports_.file != kDefaultHttpPort) {
      http_port = ports_.file;
      LOG_INFO("SendFileFirmwareOta: Using configured port %d\n", http_port);
    }
  }
  
  std::string base_url = "http://" + device_ip_ + ":" + std::to_string(http_port);
  http::Client client(base_url);
  
  // 3. Health check - if HTTP not available, fall back to simple HTTPS upload
  if (!client.health_check(3000)) {
    LOG_WARN("SendFileFirmwareOta: HTTP server not reachable, falling back to HTTPS\n");
    // Fallback to simple HTTPS upload (legacy path)
    if (!https_transfer_) {
      https_transfer_ = std::make_unique<HttpsFileTransfer>();
      std::string https_url = device_ip_ + ":" + std::to_string(ports_.file);
      if (!https_transfer_->Connect(https_url)) {
        LOG_ERROR("Failed to connect to HTTPS server\n");
        https_transfer_.reset();
        StartHeartbeat();  // Restart heartbeat on failure
        return OdinResult::kCommunicationError;
      }
    }
    FileTransferProgressCallback progress_cb = nullptr;
    if (cb) {
      OdinDeviceHandle h = handle_;
      progress_cb = [cb, h](size_t current, size_t total) {
        if (total > 0) {
          float progress = static_cast<float>(current) * 100.0f / static_cast<float>(total);
          cb(h, progress, nullptr);
        }
      };
    }
    auto result = https_transfer_->Upload(firmware_path, "firmware", progress_cb);
    if (!result.success) {
      LOG_ERROR("HTTPS upload failed: %s\n", result.error_msg.c_str());
      StartHeartbeat();  // Restart heartbeat on failure
      return OdinResult::kCommunicationError;
    }
    // OTA via HTTPS complete - device will reboot, no need to restart heartbeat
    return OdinResult::kOk;
  }
  
  // 4. Check current OTA state and reset if needed
  http::OtaStatus status;
  if (client.ota_status(status)) {
    if (status.state == "VERIFYING" || status.state == "INSTALLING_MCU" ||
        status.state == "INSTALLING_SOC" || status.state == "REBOOTING" ||
        status.state == "POST_VERIFY") {
      // OTA already in progress, wait for it
      LOG_INFO("SendFileFirmwareOta: OTA already in progress, waiting...\n");
    } else if (status.state == "UPLOADING" || status.state == "DONE" || 
               status.state == "FAILED") {
      // Need to reset first
      if (!client.ota_reset()) {
        LOG_ERROR("SendFileFirmwareOta: Failed to reset OTA state\n");
        StartHeartbeat();  // Restart heartbeat on failure
        return OdinResult::kCommunicationError;
      }
    }
  }
  
  // 5. Upload firmware (0-50% progress)
  if (status.state != "VERIFYING" && status.state != "INSTALLING_MCU" &&
      status.state != "INSTALLING_SOC" && status.state != "REBOOTING" &&
      status.state != "POST_VERIFY") {
    
    // Use static variables for progress callback (http::ProgressCallback is a function pointer)
    static OdinUpgradeProgressCallback s_cb = nullptr;
    static OdinDeviceHandle s_handle = kInvalidDeviceHandle;
    s_cb = cb;
    s_handle = handle_;
    
    auto upload_progress = [](uint64_t current, uint64_t total) -> int {
      if (s_cb && total > 0) {
        float progress = 50.0f * current / total;  // 0-50%
        s_cb(s_handle, progress, nullptr);
      }
      return 0;
    };
    
    if (!client.ota_upload(firmware_path, upload_progress)) {
      LOG_ERROR("SendFileFirmwareOta: Upload failed\n");
      StartHeartbeat();  // Restart heartbeat on failure
      return OdinResult::kCommunicationError;
    }
    
    // 6. Trigger OTA
    if (!client.ota_trigger()) {
      LOG_ERROR("SendFileFirmwareOta: Trigger failed\n");
      StartHeartbeat();  // Restart heartbeat on failure
      return OdinResult::kCommunicationError;
    }
  }
  
  // 7. Wait for completion (50-100% progress)
  auto start_time = std::chrono::steady_clock::now();
  std::string last_state;
  bool in_reboot_phase = false;
  
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    if (elapsed.count() >= kTimeoutS) {
      if (in_reboot_phase) {
        http::log_set_level(http::kLogInfo);
      }
      LOG_ERROR("SendFileFirmwareOta: Timeout waiting for completion\n");
      // No need to restart heartbeat - device is in unknown state
      return OdinResult::kTimeout;
    }
    
    if (!client.ota_status(status)) {
      // During reboot, connection may be lost temporarily - suppress mongoose errors
      if (!in_reboot_phase) {
        in_reboot_phase = true;
        http::log_set_level(http::kLogNone);
        LOG_INFO("SendFileFirmwareOta: Device rebooting, waiting for reconnection...\n");
      }
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }
    
    // Restore log level when connection is back
    if (in_reboot_phase) {
      in_reboot_phase = false;
      http::log_set_level(http::kLogInfo);
      LOG_INFO("SendFileFirmwareOta: Device reconnected\n");
    }
    
    // Log state transitions
    if (status.state != last_state) {
      LOG_INFO("SendFileFirmwareOta: State: %s\n", status.state.c_str());
      last_state = status.state;
    }
    
    // Calculate and report progress
    float progress = MapOtaStateToProgress(status.state, status.progress);
    if (progress < 0) {
      LOG_ERROR("SendFileFirmwareOta: OTA failed, error: %s\n", status.error.c_str());
      // No need to restart heartbeat - device reported failure
      return OdinResult::kUnknownError;
    }
    
    if (cb) {
      cb(handle_, progress, nullptr);
    }
    
    // Check terminal states
    if (status.state == "DONE") {
      if (cb) {
        cb(handle_, 100.0f, nullptr);
      }
      // OTA complete - device has rebooted, notify upper layer to detach and reconnect
      LOG_INFO("SendFileFirmwareOta: OTA completed successfully, notifying device offline for reconnection\n");
      if (heartbeat_failed_cb_) {
        heartbeat_failed_cb_(handle_);
      }
      return OdinResult::kOk;
    }
    
    if (status.state == "FAILED") {
      LOG_ERROR("SendFileFirmwareOta: Device reported failure: %s\n", status.error.c_str());
      // No need to restart heartbeat - device reported failure
      return OdinResult::kUnknownError;
    }
    
    // Poll interval: 2s normal, 5s during reboot
    int sleep_ms = (status.state == "REBOOTING") ? 5000 : 2000;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }
}

// =============================================================================
// Heartbeat Thread Implementation
// =============================================================================

void Odin2Device::StartHeartbeat() {
  if (heartbeat_running_.load()) return;
  
  // Skip if device doesn't support heartbeat (detected during Connect)
  if (!heartbeat_supported_) {
    LOG_INFO("Odin2Device: Heartbeat not supported, skipping heartbeat thread\n");
    return;
  }
  
  // Reset heartbeat state
  first_heartbeat_ = true;
  last_heartbeat_seq_ = 0;
  
  heartbeat_running_.store(true);
  heartbeat_thread_ = std::thread(&Odin2Device::HeartbeatThread, this);
  LOG_DEBUG("Odin2Device: Heartbeat thread started\n");
}

void Odin2Device::StopHeartbeat() {
  if (!heartbeat_running_.load()) return;
  
  heartbeat_running_.store(false);
  
  // Check if we're being called from the heartbeat thread itself
  // (e.g., during heartbeat failure callback -> DisconnectDevice -> StopHeartbeat)
  // In that case, we can't join ourselves, so detach instead
  if (heartbeat_thread_.joinable()) {
    if (std::this_thread::get_id() == heartbeat_thread_.get_id()) {
      heartbeat_thread_.detach();
      LOG_DEBUG("Odin2Device: Heartbeat thread detached (called from within)\n");
    } else {
      heartbeat_thread_.join();
      LOG_DEBUG("Odin2Device: Heartbeat thread stopped\n");
    }
  }
}

void Odin2Device::HeartbeatThread() {
  int consecutive_failures = 0;
  
  while (heartbeat_running_.load()) {

    // Sleep first, then send heartbeat
    // This allows quick shutdown without waiting for heartbeat
    uint32_t elapsed = 0;
    while (heartbeat_running_.load() && elapsed < heartbeat_interval_ms_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      elapsed += 100;
    }
    
    if (!heartbeat_running_.load()) break;
    
    // Skip heartbeat if device doesn't support it (detected during probe phase)
    if (!heartbeat_supported_) {
      continue;
    }
    
    if (SendHeartbeat()) {
      consecutive_failures = 0;
      // Probe successful - device supports heartbeat
      if (heartbeat_probing_) {
        heartbeat_probing_ = false;
        LOG_INFO("Odin2Device: Heartbeat supported, probe complete\n");
      }
    } else {
      consecutive_failures++;
      
      // Probe phase: check if device supports heartbeat
      if (heartbeat_probing_) {
        if (consecutive_failures >= kHeartbeatProbeMaxFailures) {
          LOG_WARN("Odin2Device: Heartbeat probe failed %d times, device may not support heartbeat. Disabling.\n",
                   consecutive_failures);
          heartbeat_supported_ = false;
          consecutive_failures = 0;
        }
        continue;  // Don't trigger offline during probe phase
      }
      
      // Normal phase: trigger offline after max failures
      LOG_WARN("Odin2Device: Heartbeat failed (%d/%d)\n", 
               consecutive_failures, heartbeat_max_failures_);
      
      if (consecutive_failures >= heartbeat_max_failures_) {
        LOG_ERROR("Odin2Device: Heartbeat failed %d times (%ums timeout), device offline\n",
                  consecutive_failures, heartbeat_max_failures_ * heartbeat_interval_ms_);
        // Notify upper layer about connection loss
        if (heartbeat_failed_cb_) {
          heartbeat_failed_cb_(handle_);
        }
        break;
      }
    }
  }
}

void Odin2Device::SetHeartbeatFailedCallback(HeartbeatFailedCallback cb) {
  heartbeat_failed_cb_ = cb;
}

void Odin2Device::SetHeartbeatInterval(uint32_t interval_ms) {
  heartbeat_interval_ms_ = interval_ms;
  LOG_INFO("Odin2Device: Heartbeat interval set to %u ms\n", interval_ms);
}

void Odin2Device::SetHeartbeatTimeout(uint32_t timeout_ms) {
  // Calculate max failures based on timeout and interval
  heartbeat_max_failures_ = static_cast<int>((timeout_ms + heartbeat_interval_ms_ - 1) / heartbeat_interval_ms_);
  LOG_INFO("Odin2Device: Heartbeat timeout set to %u ms, max failures = %d\n", 
           timeout_ms, heartbeat_max_failures_);
}

bool Odin2Device::SendHeartbeat() {
  // Build payload: heartbeat_interval_ms (uint16_t, little-endian)
  std::vector<uint8_t> payload(2);
  uint16_t interval_ms = static_cast<uint16_t>(heartbeat_interval_ms_);
  payload[0] = static_cast<uint8_t>(interval_ms & 0xFF);
  payload[1] = static_cast<uint8_t>((interval_ms >> 8) & 0xFF);
  
  std::lock_guard<std::mutex> lock(mutex_);
  if (!heartbeat_channel_) {
    LOG_WARN("Odin2Device::SendHeartbeat: heartbeat_channel_ is null\n");
    return false;
  }

  OdinCommandSyncResponse response;
  if (!heartbeat_channel_->SendSync(static_cast<uint16_t>(Odin2CmdId::kHeartbeat), payload,
                                    response, 1000)) {  // 1 second timeout
    LOG_WARN("Odin2Device::SendHeartbeat: SendSync failed (no response)\n");
    return false;
  }

  // Validate response: result must be OK (ret_code = 0)
  if (response.result != OdinResult::kOk) {
    LOG_WARN("Odin2Device::SendHeartbeat: response.result=%d (expected OK)\n",
             static_cast<int>(response.result));
    return false;
  }
  
  // Parse ACK payload: ret_code (uint8_t) + uptime_s (uint32_t) = 5 bytes
  const auto& ack_payload = response.response.payload;
  if (ack_payload.size() >= 5) {
    uint8_t ret_code = ack_payload[0];
    uint32_t uptime_s = static_cast<uint32_t>(ack_payload[1]) |
                        (static_cast<uint32_t>(ack_payload[2]) << 8) |
                        (static_cast<uint32_t>(ack_payload[3]) << 16) |
                        (static_cast<uint32_t>(ack_payload[4]) << 24);
    if (ret_code != 0) {
      LOG_WARN("Odin2Device::SendHeartbeat: ret_code=%u (expected 0)\n", ret_code);
      return false;
    }
    LOG_DEBUG("Odin2Device::SendHeartbeat: OK, device uptime=%u seconds\n", uptime_s);
  }

  // Validate seq: must be greater than last seq (except for first heartbeat)
  uint16_t current_seq = response.response.seq;
  if (first_heartbeat_) {
    first_heartbeat_ = false;
    last_heartbeat_seq_ = current_seq;
    LOG_DEBUG("Odin2Device::SendHeartbeat: first heartbeat, seq=%u\n", current_seq);
  } else {
    // Check if seq is incremented (handle wrap-around: 0 > 65535 is valid)
    bool seq_valid = (current_seq > last_heartbeat_seq_) || 
                     (last_heartbeat_seq_ > 65000 && current_seq < 1000);  // wrap-around
    if (!seq_valid) {
      LOG_WARN("Odin2Device::SendHeartbeat: seq not incremented, current=%u, last=%u\n",
               current_seq, last_heartbeat_seq_);
      return false;
    }
    last_heartbeat_seq_ = current_seq;
  }

  return true;
}

}  // namespace sdk
}  // namespace odin

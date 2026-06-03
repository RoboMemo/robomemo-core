#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "IDevice.h"
#include "ITransport.hpp"
#include "odin_command_channel.h"
#include "../src/SlamOdomSynchronizer.h"
#include "https/HttpsFileTransfer.hpp"
#include "OdinProtocol.hpp"

namespace odin {
namespace sdk {

// Constants for Odin2 device (uses protocol-defined constants, auto-updates when struct changes)
constexpr size_t kOdin2DataHeaderSize = OdinConst::kDataHeaderSize;
constexpr size_t kOdin2UdpMaxDataSize = 1472 - kOdin2DataHeaderSize;  // MTU - header
constexpr uint16_t kOdin2PclWidth = 256;
constexpr uint16_t kOdin2PclHeight = 192;

/**
 * @brief File information structure for file transfer
 */
#pragma pack(push, 1)
struct FileInfo {
  char filename[128];
  uint64_t filesize;
  uint8_t md5[16];
};
#pragma pack(pop)

/**
 * @brief Device port configuration
 */
struct DevicePorts {
  uint16_t raw_point = 0;
  uint16_t slam = 0;
  uint16_t jpeg = 0;
  uint16_t imu = 0;
  uint16_t odom = 0;
  uint16_t file = 0;
  uint16_t jpeg2 = 0;
};

/**
 * @brief Point cloud frame assembly state
 */
struct PointFrameState {
  OdinPointCloudPacket packet;
  uint16_t last_udp_seq = 0;
  uint16_t udp_segments = 0;
  bool active = false;
  void Reset() {
    packet = OdinPointCloudPacket();
    last_udp_seq = 0;
    udp_segments = 0;
    active = false;
  }
};

/**
 * @brief TCP stream buffer for frame reassembly
 */
struct TcpStreamBuffer {
  std::vector<uint8_t> buffer;
  void Append(const uint8_t* data, size_t len) {
    buffer.insert(buffer.end(), data, data + len);
  }
  void Clear() { buffer.clear(); }
  size_t Size() const { return buffer.size(); }
  const uint8_t* Data() const { return buffer.data(); }
  void Consume(size_t len) {
    if (len >= buffer.size()) {
      buffer.clear();
    } else {
      buffer.erase(buffer.begin(), buffer.begin() + len);
    }
  }
};

/**
 * @brief Image frame assembly state
 */
struct ImageFrameState {
  OdinImagePacket packet;
  uint16_t last_udp_seq = 0;
  uint16_t udp_segments = 0;
  uint8_t last_byte = 0;
  bool last_byte_valid = false;
  bool active = false;
  void Reset() {
    packet = OdinImagePacket();
    last_udp_seq = 0;
    udp_segments = 0;
    last_byte = 0;
    last_byte_valid = false;
    active = false;
  }
};

/**
 * @brief Odin2 device implementation (Ethernet)
 *
 * Implements IDevice interface for Odin2 series devices using UDP transport.
 */
class Odin2Device : public IDevice {
 public:
  /**
   * @brief Construct a network device
   * @param handle Device handle assigned by SDK
   */
  explicit Odin2Device(OdinDeviceHandle handle);
  ~Odin2Device() override;

  // IDevice interface implementation
  bool Connect(const DiscoveredDevice& discovered_device) override;
  void Disconnect() override;
  bool IsConnected() const override;

  OdinDeviceHandle GetHandle() const override;
  std::string GetSerialNumber() const override;
  std::string GetModel() const override;
  MTConnectionType GetConnectionType() const override;
  const DiscoveredDevice& GetDiscoveredDevice() const override;

  OdinResult GetFirmwareVersion(std::string& version, uint32_t timeout_ms = 1000) override;
  OdinResult SetOperatingMode(OdinOperatingMode mode, uint32_t timeout_ms = 1000) override;

  // Callback registration
  void RegisterPointCloudCallback(OdinPointCloudCallback cb, void* user_data) override;
  void RegisterSlamCallback(OdinSlamCallback cb, void* user_data) override;
  void RegisterImageCallback(OdinImageCallback cb, void* user_data) override;
  void RegisterImageCallback2(OdinImageCallback2 cb, void* user_data) override;
  void RegisterImuCallback(OdinImuCallback cb, void* user_data) override;
  void RegisterOdomCallback(OdinOdomCallback cb, void* user_data) override;

  // Data channel management
  void StopDataChannels() override;

  OdinResult SendFile(OdinFileType type, const std::string& file_path,
                      OdinUpgradeProgressCallback cb) override;
  OdinResult GetFile(OdinFileType type, const std::string& save_path,
                     OdinUpgradeProgressCallback cb) override;
  OdinResult SetSensorMode(uint8_t mode, uint32_t timeout_ms = 1000) override;
  OdinResult StartStream(OdinDataChannel channel, OdinTransportMode transport,
                         const OdinStreamCfg* mode = nullptr,
                         uint32_t timeout_ms = 1000) override;
  OdinResult GetSensorCapability(std::vector<OdinSensorCapability>& capabilities,
                                 const std::vector<OdinDataChannel>& channels,
                                 uint32_t timeout_ms) override;
  OdinResult CloseStream(OdinDataChannel channel, uint32_t timeout_ms) override;
  void EnableSlamOdomSync(bool enabled, uint32_t max_frame_lag = 10) override;
  void SetHeartbeatFailedCallback(HeartbeatFailedCallback cb) override;
  void SetHeartbeatInterval(uint32_t interval_ms);
  void SetHeartbeatTimeout(uint32_t timeout_ms);

  // Network-specific methods
  CommandChannel* GetCommandChannel() { return command_channel_.get(); }

 private:
  // Port generation helper
  uint16_t GenerateAvailablePort();
  bool IsPortAvailable(const std::string& ip, uint16_t port);
  std::set<uint16_t> used_ports_;  // Internal port tracking

  // Data channel handlers
  void OnDataReceived(const uint8_t* data, size_t length);
  void OnImuData(const uint8_t* data, size_t length);
  void OnFileData(const uint8_t* data, size_t length);
  
  // Firmware OTA helper (complete HTTP OTA flow)
  OdinResult SendFileFirmwareOta(const std::string& file_path, OdinUpgradeProgressCallback cb);
  
  // TCP/UDP data handlers with frame reassembly for TCP
  void OnTcpOrUdpData(OdinDataChannel channel, OdinTransportMode transport,
                      const uint8_t* data, size_t length);
  void OnTcpOrUdpImuData(OdinDataChannel channel, OdinTransportMode transport,
                         const uint8_t* data, size_t length);
  void ProcessTcpBuffer(OdinDataChannel channel, 
                        std::function<void(const uint8_t*, size_t)> handler);

  // Frame processing helpers
  void ProcessPointFragment(OdinDataChannel channel, const OdinPointCloudPacket& fragment);
  void ProcessImageFragment(OdinDataChannel channel, const OdinImagePacket& fragment);
  void EmitPointFrame(PointFrameState& state);
  void EmitImageFrame(ImageFrameState& state);
  void FillLostPacketsWithZeros(PointFrameState& state, uint16_t lost_packets,
                                size_t points_per_packet);
  bool DetectJpegTerminator(ImageFrameState& state, const std::vector<uint8_t>& chunk);
  uint16_t ExpectedPointUdpCount(OdinDataChannel channel) const;

  // Parsing helpers
  bool ParsePointCloudPacket(const uint8_t* data, size_t length, OdinPointCloudPacket* packet);
  bool ParseImagePacket(const uint8_t* data, size_t length, OdinImagePacket* packet);
  bool ParseImuPacket(const uint8_t* data, size_t length, OdinImuPacket* packet);
  bool ParseOdomPacket(const uint8_t* data, size_t length, OdinOdomPacket* packet);

  OdinDeviceHandle handle_;
  DevicePorts ports_;
  uint32_t device_ip_host_ = 0;
  std::string host_ip_;
  std::string device_ip_;

  std::string serial_number_;
  std::string model_;
  std::string firmware_version_;
  
  // Store full device info from Connect for heartbeat failure notification
  DiscoveredDevice discovered_device_;

  // Command channel (TCP 60001)
  std::unique_ptr<CommandChannel> command_channel_;
  
  // Heartbeat channel (TCP 60002) - separate from command channel
  std::unique_ptr<CommandChannel> heartbeat_channel_;

  // Data channels (device owns these) - can be UDP or TCP based on StartStream transport mode
  std::unique_ptr<ITransport> point_channel_;        // raw point cloud
  std::unique_ptr<ITransport> slam_channel_;         // slam point cloud
  std::unique_ptr<ITransport> jpeg_channel_;         // jpeg image 1
  std::unique_ptr<ITransport> jpeg2_channel_;        // jpeg image 2
  std::unique_ptr<ITransport> imu_channel_;          // imu data
  std::unique_ptr<ITransport> odom_channel_;         // odom data
  std::unique_ptr<HttpsFileTransfer> https_transfer_;  // file transfer (HTTPS)

  // Frame assembly state
  PointFrameState raw_point_state_;
  PointFrameState slam_point_state_;
  ImageFrameState image_state_;
  ImageFrameState image2_state_;
  std::mutex frame_mutex_;

  // TCP stream buffers for frame reassembly (one per data channel)
  std::map<OdinDataChannel, TcpStreamBuffer> tcp_buffers_;
  std::mutex tcp_buffer_mutex_;
  OdinTransportMode current_transport_mode_ = OdinTransportMode::kUdp;

  // Callbacks
  OdinPointCloudCallback point_callback_ = nullptr;
  void* point_user_data_ = nullptr;
  OdinSlamCallback slam_callback_ = nullptr;
  void* slam_user_data_ = nullptr;
  OdinImageCallback image_callback_ = nullptr;
  void* image_user_data_ = nullptr;
  OdinImageCallback2 image_callback_2_ = nullptr;
  void* image_user_data_2_ = nullptr;
  OdinImuCallback imu_callback_ = nullptr;
  void* imu_user_data_ = nullptr;
  OdinOdomCallback odom_callback_ = nullptr;
  void* odom_user_data_ = nullptr;

  std::atomic<bool> connected_{false};
  mutable std::mutex mutex_;
  std::mutex callback_mutex_;

  // SLAM-Odom synchronization (per-device instance)
  std::unique_ptr<SlamOdomSynchronizer> slam_odom_sync_;

  // Internal capability table: channel -> list of configs (index = resolution_id)
  // Each config contains width, height, fps array. fps_id is index within fps array.
  std::map<OdinDataChannel, std::vector<OdinStreamCfg>> capability_table_;
  std::mutex capability_mutex_;

  // File transfer state
  std::atomic<bool> file_transfer_active_{false};
  FileInfo file_info_;
  std::string save_path_;
  uint64_t bytes_transferred_{0};
  FILE* save_file_fd_{nullptr};
  std::mutex file_mutex_;

  // Heartbeat thread for connection monitoring
  std::thread heartbeat_thread_;
  std::atomic<bool> heartbeat_running_{false};
  uint32_t heartbeat_interval_ms_ = 500;   // 500ms interval
  int heartbeat_max_failures_ = 6;         // 3 seconds total (6 * 500ms)
  HeartbeatFailedCallback heartbeat_failed_cb_;
  uint16_t last_heartbeat_seq_ = 0;        // Last received heartbeat seq for validation
  bool first_heartbeat_ = true;            // Flag for first heartbeat (no seq check)
  
  // Heartbeat auto-detection for legacy device compatibility
  bool heartbeat_supported_ = true;        // Auto-detect: disable if device doesn't support heartbeat
  bool heartbeat_probing_ = true;          // True during probe phase
  static constexpr int kHeartbeatProbeMaxFailures = 3;  // Probe phase max failures before disabling
  
  void StartHeartbeat();
  void StopHeartbeat();
  void HeartbeatThread();
  bool SendHeartbeat();
};

}  // namespace sdk
}  // namespace odin

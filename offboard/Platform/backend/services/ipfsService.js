/**
 * ipfsService.js
 * ─────────────────────────────────────────────────────────────────────────────
 * 本地 IPFS 模拟服务。
 *
 * 生产环境可替换为:
 *   - Kubo (go-ipfs) HTTP API: http://localhost:5001
 *   - Pinata API: https://api.pinata.cloud
 *   - web3.storage
 *
 * 本地模拟方案:
 *   - 文件存储到 uploads/ipfs/<cid>/
 *   - CID 用 SHA-256 哈希前缀模拟 (Qm...)
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const IPFS_STORE = path.join(__dirname, "..", "uploads", "ipfs");
fs.mkdirSync(IPFS_STORE, { recursive: true });

/**
 * 生成模拟 CID (类 IPFS Qm... 格式)
 */
function generateCID(buffer) {
  const hash = crypto.createHash("sha256").update(buffer).digest("hex");
  // 模拟 base58 CID: "Qm" + hex[:44]
  return "Qm" + hash.substring(0, 44);
}

/**
 * 上传 Buffer/文件到本地 IPFS 模拟存储
 * @param {Buffer|string} content - 文件内容或文件路径
 * @param {string} filename - 原始文件名 (用于 metadata)
 * @returns {{ cid: string, url: string, size: number }}
 */
async function uploadToIPFS(content, filename) {
  let buffer;
  if (typeof content === "string" && fs.existsSync(content)) {
    buffer = fs.readFileSync(content);
  } else if (Buffer.isBuffer(content)) {
    buffer = content;
  } else {
    buffer = Buffer.from(content, "utf-8");
  }

  const cid = generateCID(buffer);
  const dir = path.join(IPFS_STORE, cid);
  fs.mkdirSync(dir, { recursive: true });

  const filePath = path.join(dir, filename);
  fs.writeFileSync(filePath, buffer);

  // 写入 meta
  fs.writeFileSync(
    path.join(dir, "_meta.json"),
    JSON.stringify({
      cid,
      filename,
      size: buffer.length,
      uploadedAt: new Date().toISOString(),
    })
  );

  return {
    cid,
    url: `/ipfs/${cid}/${filename}`,
    gatewayUrl: `http://localhost:3001/ipfs/${cid}/${filename}`,
    size: buffer.length,
  };
}

/**
 * 上传 JSON 对象到本地 IPFS
 */
async function uploadJSONToIPFS(obj, filename = "metadata.json") {
  const content = JSON.stringify(obj, null, 2);
  return uploadToIPFS(Buffer.from(content, "utf-8"), filename);
}

/**
 * 获取 IPFS 内容 (本地读取)
 */
function getFromIPFS(cid, filename) {
  const filePath = path.join(IPFS_STORE, cid, filename);
  if (!fs.existsSync(filePath)) return null;
  return fs.readFileSync(filePath);
}

/**
 * 列出本地 IPFS 存储的所有 CID
 */
function listIPFSContent() {
  if (!fs.existsSync(IPFS_STORE)) return [];
  return fs.readdirSync(IPFS_STORE).filter((d) => {
    const metaPath = path.join(IPFS_STORE, d, "_meta.json");
    return fs.existsSync(metaPath);
  });
}

/**
 * 读取 IPFS CID 的 meta 信息
 */
function getIPFSMeta(cid) {
  const metaPath = path.join(IPFS_STORE, cid, "_meta.json");
  if (!fs.existsSync(metaPath)) return null;
  return JSON.parse(fs.readFileSync(metaPath, "utf-8"));
}

module.exports = {
  uploadToIPFS,
  uploadJSONToIPFS,
  getFromIPFS,
  listIPFSContent,
  getIPFSMeta,
  IPFS_STORE,
};

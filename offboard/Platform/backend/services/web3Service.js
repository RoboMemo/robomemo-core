/**
 * web3Service.js
 * ─────────────────────────────────────────────────────────────────────────────
 * 后端 Web3 服务：通过 ethers.js 与本地 Hardhat 节点上的智能合约交互。
 *
 * 职责:
 *   - 读取 web3-deployment.json 获取合约地址和 ABI
 *   - 提供 mintNFT / listForSale / buyItem / getListings 等操作
 *   - 后端使用 deployer 账户签名（仅限 mint），其余操作由前端钱包签名
 */

const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

const DEPLOYMENT_PATH = path.join(__dirname, "..", "web3-deployment.json");
const RPC_URL = process.env.WEB3_RPC_URL || "http://127.0.0.1:8545";

// ─── Provider / Signer ────────────────────────────────────────────────────────

let _provider = null;
let _deployment = null;

function getProvider() {
  if (!_provider) {
    _provider = new ethers.JsonRpcProvider(RPC_URL);
  }
  return _provider;
}

function getDeployment() {
  if (!_deployment) {
    if (!fs.existsSync(DEPLOYMENT_PATH)) {
      throw new Error(
        "web3-deployment.json not found. Run: cd contracts && npx hardhat run scripts/deploy.js --network localhost"
      );
    }
    _deployment = JSON.parse(fs.readFileSync(DEPLOYMENT_PATH, "utf-8"));
  }
  return _deployment;
}

function getNFTContract(signerOrProvider) {
  const dep = getDeployment();
  return new ethers.Contract(
    dep.contracts.RoboDataNFT.address,
    dep.contracts.RoboDataNFT.abi,
    signerOrProvider || getProvider()
  );
}

function getMarketContract(signerOrProvider) {
  const dep = getDeployment();
  return new ethers.Contract(
    dep.contracts.RoboDataMarketplace.address,
    dep.contracts.RoboDataMarketplace.abi,
    signerOrProvider || getProvider()
  );
}

/**
 * 返回后端部署者账户（用于代理 mint，前端没有私钥时使用）
 * 在本地 Hardhat 中使用账户 #0
 */
async function getDeployerSigner() {
  const privateKey = process.env.DEPLOYER_PRIVATE_KEY;
  if (privateKey) {
    return new ethers.Wallet(privateKey, getProvider());
  }
  // 本地 Hardhat 默认账户 #0
  const provider = getProvider();
  const accounts = await provider.listAccounts();
  if (accounts.length === 0) throw new Error("No Hardhat accounts available");
  return await provider.getSigner(accounts[0].address);
}

// ─── NFT Operations ───────────────────────────────────────────────────────────

/**
 * 后端代理铸造 NFT（由 deployer 签名，mint 给指定地址）
 */
async function mintDatasetNFT({ toAddress, title, robotType, taskName, videoCID, sftCID, metaCID, tokenURI }) {
  const signer = await getDeployerSigner();
  const nft = getNFTContract(signer);

  const tx = await nft.mintDataset(
    toAddress,
    title,
    robotType,
    taskName,
    videoCID,
    sftCID,
    metaCID,
    tokenURI
  );
  const receipt = await tx.wait();

  // 从事件中提取 tokenId
  const event = receipt.logs
    .map((log) => {
      try { return nft.interface.parseLog(log); } catch { return null; }
    })
    .find((e) => e && e.name === "DatasetMinted");

  const tokenId = event ? event.args.tokenId.toString() : null;
  return { txHash: receipt.hash, tokenId, blockNumber: receipt.blockNumber };
}

/**
 * 获取 NFT 元数据
 */
async function getNFTMeta(tokenId) {
  const nft = getNFTContract();
  const meta = await nft.getDatasetMeta(tokenId);
  const owner = await nft.ownerOf(tokenId);
  const uri = await nft.tokenURI(tokenId);
  return {
    tokenId: tokenId.toString(),
    owner,
    tokenURI: uri,
    title: meta.title,
    robotType: meta.robotType,
    taskName: meta.taskName,
    videoCID: meta.videoCID,
    sftCID: meta.sftCID,
    metaCID: meta.metaCID,
    mintedAt: Number(meta.mintedAt),
    creator: meta.creator,
  };
}

/**
 * 获取某地址持有的所有 tokenId
 */
async function getOwnedTokens(address) {
  const nft = getNFTContract();
  const total = await nft.totalSupply();
  const owned = [];
  for (let i = 1; i <= Number(total); i++) {
    try {
      const owner = await nft.ownerOf(i);
      if (owner.toLowerCase() === address.toLowerCase()) {
        owned.push(i);
      }
    } catch {
      // token burned or not minted
    }
  }
  return owned;
}

// ─── Marketplace Operations ───────────────────────────────────────────────────

/**
 * 获取所有活跃上架列表
 */
async function getActiveListings() {
  const market = getMarketContract();
  const nft = getNFTContract();
  const listings = await market.getActiveListings();

  // 附加 NFT 元数据
  return Promise.all(
    listings.map(async (l) => {
      let meta = null;
      try { meta = await nft.getDatasetMeta(l.tokenId); } catch {}
      return {
        tokenId: l.tokenId.toString(),
        seller: l.seller,
        price: l.price.toString(),
        priceETH: ethers.formatEther(l.price),
        active: l.active,
        listedAt: Number(l.listedAt),
        meta: meta
          ? {
              title: meta.title,
              robotType: meta.robotType,
              taskName: meta.taskName,
              videoCID: meta.videoCID,
              sftCID: meta.sftCID,
              creator: meta.creator,
            }
          : null,
      };
    })
  );
}

/**
 * 获取合约部署信息（前端需要）
 */
function getContractInfo() {
  const dep = getDeployment();
  return {
    network: dep.network,
    chainId: dep.chainId,
    rpcUrl: RPC_URL,
    contracts: {
      RoboDataNFT: {
        address: dep.contracts.RoboDataNFT.address,
        abi: dep.contracts.RoboDataNFT.abi,
      },
      RoboDataMarketplace: {
        address: dep.contracts.RoboDataMarketplace.address,
        abi: dep.contracts.RoboDataMarketplace.abi,
      },
    },
  };
}

/**
 * 检查 Web3 连接状态
 */
async function getWeb3Status() {
  try {
    const provider = getProvider();
    const blockNumber = await provider.getBlockNumber();
    const network = await provider.getNetwork();
    const dep = getDeployment();
    return {
      connected: true,
      blockNumber,
      chainId: network.chainId.toString(),
      rpcUrl: RPC_URL,
      contracts: {
        nft: dep.contracts.RoboDataNFT.address,
        marketplace: dep.contracts.RoboDataMarketplace.address,
      },
    };
  } catch (err) {
    return { connected: false, error: err.message };
  }
}

module.exports = {
  getProvider,
  getNFTContract,
  getMarketContract,
  mintDatasetNFT,
  getNFTMeta,
  getOwnedTokens,
  getActiveListings,
  getContractInfo,
  getWeb3Status,
};

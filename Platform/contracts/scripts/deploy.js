const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with:", deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address)), "ETH");

  // 1. Deploy RoboDataNFT
  const NFT = await hre.ethers.getContractFactory("RoboDataNFT");
  const nft = await NFT.deploy();
  await nft.waitForDeployment();
  const nftAddress = await nft.getAddress();
  console.log("RoboDataNFT deployed to:", nftAddress);

  // 2. Deploy RoboDataMarketplace
  const Market = await hre.ethers.getContractFactory("RoboDataMarketplace");
  const market = await Market.deploy(nftAddress);
  await market.waitForDeployment();
  const marketAddress = await market.getAddress();
  console.log("RoboDataMarketplace deployed to:", marketAddress);

  // 3. Save addresses + ABIs for backend/frontend consumption
  const artifact_nft = await hre.artifacts.readArtifact("RoboDataNFT");
  const artifact_market = await hre.artifacts.readArtifact("RoboDataMarketplace");

  const deployment = {
    network: hre.network.name,
    chainId: 31337,
    deployedAt: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      RoboDataNFT: {
        address: nftAddress,
        abi: artifact_nft.abi,
      },
      RoboDataMarketplace: {
        address: marketAddress,
        abi: artifact_market.abi,
      },
    },
  };

  // Write to contracts/deployment.json (used by backend)
  const deployPath = path.join(__dirname, "..", "deployment.json");
  fs.writeFileSync(deployPath, JSON.stringify(deployment, null, 2));
  console.log("Deployment info saved to:", deployPath);

  // Also write to backend/web3-deployment.json for easy access
  const backendPath = path.join(__dirname, "..", "..", "backend", "web3-deployment.json");
  fs.writeFileSync(backendPath, JSON.stringify(deployment, null, 2));
  console.log("Deployment info also saved to:", backendPath);

  console.log("\n=== Deployment Complete ===");
  console.log("RoboDataNFT:        ", nftAddress);
  console.log("RoboDataMarketplace:", marketAddress);
  console.log("\nTo start the local node: cd contracts && npx hardhat node");
  console.log("To deploy:            cd contracts && npx hardhat run scripts/deploy.js --network localhost");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

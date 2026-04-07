const { expect } = require("chai");
const hre = require("hardhat");

describe("RoboDataNFT + RoboDataMarketplace", function () {
  let nft, market, owner, seller, buyer;

  beforeEach(async () => {
    [owner, seller, buyer] = await hre.ethers.getSigners();

    const NFT = await hre.ethers.getContractFactory("RoboDataNFT");
    nft = await NFT.deploy();

    const Market = await hre.ethers.getContractFactory("RoboDataMarketplace");
    market = await Market.deploy(await nft.getAddress());
  });

  it("should mint a dataset NFT", async () => {
    await nft.connect(owner).mintDataset(
      seller.address,
      "Screw Fastening Demo",
      "UR5",
      "screw_fastening",
      "Qm_video_cid",
      "Qm_sft_cid",
      "Qm_meta_cid",
      "ipfs://Qm_meta_cid"
    );
    expect(await nft.ownerOf(1)).to.equal(seller.address);
    expect(await nft.totalSupply()).to.equal(1);

    const meta = await nft.getDatasetMeta(1);
    expect(meta.title).to.equal("Screw Fastening Demo");
    expect(meta.videoCID).to.equal("Qm_video_cid");
  });

  it("should list and buy an NFT", async () => {
    // Mint to seller
    await nft.connect(owner).mintDataset(
      seller.address, "Demo", "Franka", "pick_place",
      "vid_cid", "sft_cid", "meta_cid", "ipfs://meta_cid"
    );

    const price = hre.ethers.parseEther("0.1");
    const marketAddress = await market.getAddress();

    // Seller approves marketplace
    await nft.connect(seller).approve(marketAddress, 1);

    // Seller lists
    await market.connect(seller).listItem(1, price);
    const listing = await market.getListing(1);
    expect(listing.active).to.be.true;
    expect(listing.price).to.equal(price);

    // Buyer buys
    const sellerBefore = await hre.ethers.provider.getBalance(seller.address);
    await market.connect(buyer).buyItem(1, { value: price });

    // NFT transferred to buyer
    expect(await nft.ownerOf(1)).to.equal(buyer.address);

    // Listing deactivated
    const listingAfter = await market.getListing(1);
    expect(listingAfter.active).to.be.false;

    // Seller received funds minus platform fee (2.5%)
    const fee = (price * 250n) / 10000n;
    const expected = price - fee;
    const sellerAfter = await hre.ethers.provider.getBalance(seller.address);
    expect(sellerAfter - sellerBefore).to.be.closeTo(expected, hre.ethers.parseEther("0.001"));
  });

  it("should delist an NFT", async () => {
    await nft.connect(owner).mintDataset(
      seller.address, "Demo", "Franka", "pick_place",
      "vid_cid", "sft_cid", "meta_cid", "ipfs://meta_cid"
    );
    const price = hre.ethers.parseEther("0.1");
    await nft.connect(seller).approve(await market.getAddress(), 1);
    await market.connect(seller).listItem(1, price);

    await market.connect(seller).delistItem(1);
    const listing = await market.getListing(1);
    expect(listing.active).to.be.false;
  });

  it("should return active listings", async () => {
    // Mint 2 tokens
    for (let i = 0; i < 2; i++) {
      await nft.connect(owner).mintDataset(
        seller.address, `Demo ${i}`, "Franka", "task",
        `vid_${i}`, `sft_${i}`, `meta_${i}`, `ipfs://meta_${i}`
      );
    }
    const price = hre.ethers.parseEther("0.1");
    await nft.connect(seller).setApprovalForAll(await market.getAddress(), true);
    await market.connect(seller).listItem(1, price);
    await market.connect(seller).listItem(2, price);

    const active = await market.getActiveListings();
    expect(active.length).to.equal(2);
  });
});

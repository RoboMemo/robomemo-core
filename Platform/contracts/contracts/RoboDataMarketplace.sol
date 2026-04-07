// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title RoboDataMarketplace
 * @dev Fixed-price marketplace for trading RoboDataNFT tokens.
 *      Sellers list tokens at a price in wei; buyers pay directly.
 *      Platform collects a configurable fee (default 2.5%).
 */
contract RoboDataMarketplace is ReentrancyGuard, Ownable {
    // ─── Constants ───────────────────────────────────────────────────────────

    uint256 public constant FEE_DENOMINATOR = 10000;
    uint256 public platformFeeBps = 250; // 2.5%

    // ─── Structs ─────────────────────────────────────────────────────────────

    struct Listing {
        uint256 tokenId;
        address seller;
        uint256 price;      // in wei
        bool active;
        uint256 listedAt;
    }

    // ─── State ───────────────────────────────────────────────────────────────

    IERC721 public immutable nftContract;

    /// tokenId → Listing
    mapping(uint256 => Listing) public listings;

    /// All ever-listed tokenIds (for enumeration)
    uint256[] public listedTokenIds;

    /// Accumulated platform fees (withdrawable by owner)
    uint256 public accumulatedFees;

    // ─── Events ──────────────────────────────────────────────────────────────

    event Listed(uint256 indexed tokenId, address indexed seller, uint256 price);
    event Delisted(uint256 indexed tokenId, address indexed seller);
    event Sold(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price);
    event PriceUpdated(uint256 indexed tokenId, uint256 oldPrice, uint256 newPrice);

    // ─── Constructor ─────────────────────────────────────────────────────────

    constructor(address nftAddress) Ownable(msg.sender) {
        nftContract = IERC721(nftAddress);
    }

    // ─── Seller actions ──────────────────────────────────────────────────────

    /**
     * @notice List an NFT for sale.
     *         Caller must first approve this contract via nftContract.approve().
     */
    function listItem(uint256 tokenId, uint256 price) external {
        require(price > 0, "Price must be > 0");
        require(nftContract.ownerOf(tokenId) == msg.sender, "Not token owner");
        require(
            nftContract.getApproved(tokenId) == address(this) ||
            nftContract.isApprovedForAll(msg.sender, address(this)),
            "Marketplace not approved"
        );
        require(!listings[tokenId].active, "Already listed");

        listings[tokenId] = Listing({
            tokenId: tokenId,
            seller: msg.sender,
            price: price,
            active: true,
            listedAt: block.timestamp
        });
        listedTokenIds.push(tokenId);

        emit Listed(tokenId, msg.sender, price);
    }

    /**
     * @notice Update the price of an active listing.
     */
    function updatePrice(uint256 tokenId, uint256 newPrice) external {
        require(newPrice > 0, "Price must be > 0");
        Listing storage l = listings[tokenId];
        require(l.active, "Not listed");
        require(l.seller == msg.sender, "Not seller");

        uint256 old = l.price;
        l.price = newPrice;
        emit PriceUpdated(tokenId, old, newPrice);
    }

    /**
     * @notice Remove an NFT from the marketplace.
     */
    function delistItem(uint256 tokenId) external {
        Listing storage l = listings[tokenId];
        require(l.active, "Not listed");
        require(l.seller == msg.sender, "Not seller");

        l.active = false;
        emit Delisted(tokenId, msg.sender);
    }

    // ─── Buyer actions ───────────────────────────────────────────────────────

    /**
     * @notice Buy a listed NFT. Send exact price in msg.value.
     */
    function buyItem(uint256 tokenId) external payable nonReentrant {
        Listing storage l = listings[tokenId];
        require(l.active, "Not listed");
        require(msg.value == l.price, "Incorrect ETH amount");
        require(msg.sender != l.seller, "Seller cannot buy own item");

        l.active = false;

        // Calculate fees
        uint256 fee = (l.price * platformFeeBps) / FEE_DENOMINATOR;
        uint256 sellerProceeds = l.price - fee;
        accumulatedFees += fee;

        // Transfer NFT to buyer
        nftContract.safeTransferFrom(l.seller, msg.sender, tokenId);

        // Pay seller
        (bool ok, ) = payable(l.seller).call{value: sellerProceeds}("");
        require(ok, "Transfer to seller failed");

        emit Sold(tokenId, l.seller, msg.sender, l.price);
    }

    // ─── Views ───────────────────────────────────────────────────────────────

    /**
     * @notice Get all currently active listings.
     */
    function getActiveListings()
        external view
        returns (Listing[] memory result)
    {
        uint256 count = 0;
        for (uint256 i = 0; i < listedTokenIds.length; i++) {
            if (listings[listedTokenIds[i]].active) count++;
        }

        result = new Listing[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < listedTokenIds.length; i++) {
            uint256 tid = listedTokenIds[i];
            if (listings[tid].active) {
                result[idx++] = listings[tid];
            }
        }
    }

    function getListing(uint256 tokenId) external view returns (Listing memory) {
        return listings[tokenId];
    }

    // ─── Admin ───────────────────────────────────────────────────────────────

    function setPlatformFee(uint256 bps) external onlyOwner {
        require(bps <= 1000, "Fee too high (max 10%)");
        platformFeeBps = bps;
    }

    function withdrawFees() external onlyOwner {
        uint256 amount = accumulatedFees;
        accumulatedFees = 0;
        (bool ok, ) = payable(owner()).call{value: amount}("");
        require(ok, "Withdraw failed");
    }
}

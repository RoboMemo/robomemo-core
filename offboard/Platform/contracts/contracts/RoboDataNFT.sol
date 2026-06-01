// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title RoboDataNFT
 * @dev ERC-721 NFT representing ownership of a robotics dataset entry.
 *      Metadata (video CID, SFT output CID, annotations) stored on IPFS,
 *      referenced via tokenURI.
 */
contract RoboDataNFT is ERC721, ERC721URIStorage, Ownable {
    // ─── State ───────────────────────────────────────────────────────────────

    uint256 private _nextTokenId;

    /// @dev On-chain metadata for each token (subset for quick display)
    struct DatasetMeta {
        string title;
        string robotType;
        string taskName;
        string videoCID;    // IPFS CID of the raw video
        string sftCID;      // IPFS CID of SFT output (labels.jsonl + configs)
        string metaCID;     // IPFS CID of full metadata JSON
        uint256 mintedAt;
        address creator;
    }

    mapping(uint256 => DatasetMeta) public datasetMeta;

    // ─── Events ──────────────────────────────────────────────────────────────

    event DatasetMinted(
        uint256 indexed tokenId,
        address indexed creator,
        string title,
        string videoCID,
        string sftCID
    );

    // ─── Constructor ─────────────────────────────────────────────────────────

    constructor() ERC721("RoboDataNFT", "RDATA") Ownable(msg.sender) {
        _nextTokenId = 1;
    }

    // ─── Minting ─────────────────────────────────────────────────────────────

    /**
     * @notice Mint a new dataset NFT.
     * @param to         Recipient address (dataset uploader)
     * @param title      Human-readable dataset title
     * @param robotType  Robot type (e.g. "UR5", "Franka", "Unitree H1")
     * @param taskName   Task description (e.g. "screw_fastening")
     * @param videoCID   IPFS CID of the video file
     * @param sftCID     IPFS CID of the SFT annotation bundle
     * @param metaCID    IPFS CID of the full JSON metadata
     * @param tokenURI_  Full IPFS URI for the metadata JSON (ipfs://...)
     */
    function mintDataset(
        address to,
        string memory title,
        string memory robotType,
        string memory taskName,
        string memory videoCID,
        string memory sftCID,
        string memory metaCID,
        string memory tokenURI_
    ) external returns (uint256) {
        uint256 tokenId = _nextTokenId++;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI_);

        datasetMeta[tokenId] = DatasetMeta({
            title: title,
            robotType: robotType,
            taskName: taskName,
            videoCID: videoCID,
            sftCID: sftCID,
            metaCID: metaCID,
            mintedAt: block.timestamp,
            creator: to
        });

        emit DatasetMinted(tokenId, to, title, videoCID, sftCID);
        return tokenId;
    }

    // ─── Getters ─────────────────────────────────────────────────────────────

    function totalSupply() external view returns (uint256) {
        return _nextTokenId - 1;
    }

    function getDatasetMeta(uint256 tokenId) external view returns (DatasetMeta memory) {
        require(_ownerOf(tokenId) != address(0), "Token does not exist");
        return datasetMeta[tokenId];
    }

    // ─── Overrides ───────────────────────────────────────────────────────────

    function tokenURI(uint256 tokenId)
        public view override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public view override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}

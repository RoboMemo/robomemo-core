import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import {
  Upload, ShoppingCart, Wallet, RefreshCw,
  Tag, Package, CheckCircle, AlertCircle, Download,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';

const API = 'http://localhost:3001';

// ─── Types ───────────────────────────────────────────────────────────────────

interface NFTMeta {
  title: string;
  robotType: string;
  taskName: string;
  videoCID: string;
  sftCID: string;
  creator: string;
}

interface Listing {
  tokenId: string;
  seller: string;
  price: string;
  priceETH: string;
  active: boolean;
  listedAt: number;
  meta: NFTMeta | null;
}

interface Web3Status {
  connected: boolean;
  blockNumber?: number;
  chainId?: string;
  rpcUrl?: string;
  contracts?: { nft: string; marketplace: string };
  error?: string;
}

interface ContractInfo {
  network: string;
  chainId: number;
  contracts: {
    RoboDataNFT: { address: string; abi: unknown[] };
    RoboDataMarketplace: { address: string; abi: unknown[] };
  };
}

// ─── Helper ──────────────────────────────────────────────────────────────────

function shortAddr(addr: string) {
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function Web3Marketplace() {
  const [account, setAccount] = useState('');
  const [web3Status, setWeb3Status] = useState<Web3Status | null>(null);
  const [contractInfo, setContractInfo] = useState<ContractInfo | null>(null);
  const [listings, setListings] = useState<Listing[]>([]);

  // Upload form
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [sftFile, setSftFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [robotType, setRobotType] = useState('');
  const [taskName, setTaskName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [mintResult, setMintResult] = useState<Record<string, string> | null>(null);

  // List for sale
  const [listTokenId, setListTokenId] = useState('');
  const [listPrice, setListPrice] = useState('');
  const [txStatus, setTxStatus] = useState('');

  useEffect(() => {
    fetchStatus();
    fetchListings();
  }, []);

  // ── Data fetchers ──────────────────────────────────────────────────────────

  async function fetchStatus() {
    try {
      const r = await fetch(`${API}/api/web3/status`);
      const data: Web3Status = await r.json();
      setWeb3Status(data);
      if (data.connected) {
        const cr = await fetch(`${API}/api/web3/contracts`);
        if (cr.ok) setContractInfo(await cr.json());
      }
    } catch {
      setWeb3Status({ connected: false, error: '无法连接后端' });
    }
  }

  async function fetchListings() {
    try {
      const r = await fetch(`${API}/api/web3/listings`);
      if (r.ok) setListings(await r.json());
    } catch {}
  }

  // ── Wallet ────────────────────────────────────────────────────────────────

  async function connectWallet() {
    const eth = (window as unknown as { ethereum?: { request: (a: unknown) => Promise<string[]> } }).ethereum;
    if (!eth) {
      alert('请安装 MetaMask 后再使用 Web3 功能');
      return;
    }
    const accounts = await eth.request({ method: 'eth_requestAccounts' });
    setAccount(accounts[0]);
    // Try to switch to Hardhat local network (chainId 31337 = 0x7a69)
    try {
      await (eth as unknown as { request: (a: unknown) => Promise<void> }).request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0x7a69' }],
      });
    } catch (err: unknown) {
      if ((err as { code?: number }).code === 4902) {
        await (eth as unknown as { request: (a: unknown) => Promise<void> }).request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: '0x7a69',
            chainName: 'Hardhat Local',
            rpcUrls: ['http://127.0.0.1:8545'],
            nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
          }],
        });
      }
    }
  }

  // ── Upload & Mint ─────────────────────────────────────────────────────────

  async function handleUploadMint() {
    if (!account) return alert('请先连接钱包');
    if (!title || !robotType || !taskName) return alert('请填写所有必填字段');

    const form = new FormData();
    form.append('title', title);
    form.append('robotType', robotType);
    form.append('taskName', taskName);
    form.append('toAddress', account);
    if (videoFile) form.append('video', videoFile);
    if (sftFile) form.append('sft', sftFile);

    setUploading(true);
    setUploadProgress(10);
    setUploadStatus('上传文件到本地 IPFS...');
    setMintResult(null);

    try {
      setUploadProgress(40);
      const res = await fetch(`${API}/api/web3/upload`, { method: 'POST', body: form });
      setUploadProgress(80);
      setUploadStatus('在区块链上铸造 NFT...');
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setUploadProgress(100);
      setUploadStatus('NFT 铸造成功!');
      setMintResult(data);
      fetchListings();
    } catch (err: unknown) {
      setUploadStatus(`错误: ${(err as Error).message}`);
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  }

  // ── List for Sale ─────────────────────────────────────────────────────────

  async function handleListForSale() {
    if (!account || !contractInfo) return alert('请先连接钱包');
    if (!listTokenId || !listPrice) return alert('请填写 Token ID 和售价');

    setTxStatus('等待 MetaMask 授权...');
    try {
      const provider = new ethers.BrowserProvider(
        (window as unknown as { ethereum: ethers.Eip1193Provider }).ethereum
      );
      const signer = await provider.getSigner();

      const nft = new ethers.Contract(
        contractInfo.contracts.RoboDataNFT.address,
        contractInfo.contracts.RoboDataNFT.abi,
        signer
      );
      const market = new ethers.Contract(
        contractInfo.contracts.RoboDataMarketplace.address,
        contractInfo.contracts.RoboDataMarketplace.abi,
        signer
      );

      setTxStatus('步骤 1/2: 授权市场合约...');
      const approveTx = await nft.approve(
        contractInfo.contracts.RoboDataMarketplace.address,
        listTokenId
      );
      await approveTx.wait();

      setTxStatus('步骤 2/2: 上架 NFT...');
      const priceWei = ethers.parseEther(listPrice);
      const listTx = await market.listItem(listTokenId, priceWei);
      await listTx.wait();

      setTxStatus(`Token #${listTokenId} 已成功上架，售价 ${listPrice} ETH`);
      setListTokenId('');
      setListPrice('');
      fetchListings();
    } catch (err: unknown) {
      setTxStatus(`错误: ${(err as Error).message}`);
    }
  }

  // ── Buy ───────────────────────────────────────────────────────────────────

  async function handleBuy(listing: Listing) {
    if (!account || !contractInfo) return alert('请先连接钱包');

    try {
      const provider = new ethers.BrowserProvider(
        (window as unknown as { ethereum: ethers.Eip1193Provider }).ethereum
      );
      const signer = await provider.getSigner();
      const market = new ethers.Contract(
        contractInfo.contracts.RoboDataMarketplace.address,
        contractInfo.contracts.RoboDataMarketplace.abi,
        signer
      );

      const tx = await market.buyItem(listing.tokenId, { value: BigInt(listing.price) });
      await tx.wait();
      alert(`成功购买 Token #${listing.tokenId}!`);
      fetchListings();
    } catch (err: unknown) {
      alert(`购买失败: ${(err as Error).message}`);
    }
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold">Web3 机器人数据市场</h2>
          <p className="text-muted-foreground text-sm mt-1">
            将 SFT 标注数据 + 视频上链成 NFT，供他人查看、购买、下载
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={() => { fetchStatus(); fetchListings(); }}>
            <RefreshCw className="w-4 h-4 mr-1" /> 刷新
          </Button>
          {account ? (
            <Badge variant="secondary" className="font-mono text-xs px-3 py-1">
              <Wallet className="w-3 h-3 mr-1" />
              {shortAddr(account)}
            </Badge>
          ) : (
            <Button size="sm" onClick={connectWallet}>
              <Wallet className="w-4 h-4 mr-2" /> 连接 MetaMask
            </Button>
          )}
        </div>
      </div>

      {/* Chain Status Bar */}
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center flex-wrap gap-4 text-sm">
            {web3Status?.connected ? (
              <>
                <span className="flex items-center gap-1 text-green-600 font-medium">
                  <CheckCircle className="w-4 h-4" /> Hardhat 本地链已连接
                </span>
                <span className="text-muted-foreground">区块: #{web3Status.blockNumber}</span>
                <span className="text-muted-foreground">ChainId: {web3Status.chainId}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  NFT 合约: {web3Status.contracts?.nft ? shortAddr(web3Status.contracts.nft) : '—'}
                </span>
                <span className="font-mono text-xs text-muted-foreground">
                  市场合约: {web3Status.contracts?.marketplace ? shortAddr(web3Status.contracts.marketplace) : '—'}
                </span>
              </>
            ) : (
              <span className="flex items-center gap-2 text-amber-600">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>
                  {web3Status?.error || '区块链未连接 — 请先运行: cd Platform/contracts && npm run node && npm run deploy:local'}
                </span>
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Main Tabs */}
      <Tabs defaultValue="marketplace">
        <TabsList className="grid grid-cols-3 w-full max-w-md">
          <TabsTrigger value="marketplace">
            <ShoppingCart className="w-4 h-4 mr-2" /> 市场
          </TabsTrigger>
          <TabsTrigger value="upload">
            <Upload className="w-4 h-4 mr-2" /> 上传铸造
          </TabsTrigger>
          <TabsTrigger value="sell">
            <Tag className="w-4 h-4 mr-2" /> 上架出售
          </TabsTrigger>
        </TabsList>

        {/* ── Tab: Marketplace ── */}
        <TabsContent value="marketplace" className="mt-4 space-y-4">
          {listings.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center py-16 text-muted-foreground">
                <Package className="w-14 h-14 mb-4 opacity-25" />
                <p className="font-medium">暂无在售数据集</p>
                <p className="text-xs mt-1">先上传并铸造 NFT，再切换到「上架出售」标签</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {listings.map((l) => (
                <Card key={l.tokenId} className="overflow-hidden hover:shadow-md transition-shadow">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between gap-2">
                      <CardTitle className="text-sm font-semibold leading-tight">
                        {l.meta?.title || `Dataset #${l.tokenId}`}
                      </CardTitle>
                      <Badge variant="outline" className="shrink-0">#{l.tokenId}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {l.meta && (
                      <div className="text-xs space-y-1">
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-14">机器人</span>
                          <span className="font-medium">{l.meta.robotType}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-14">任务</span>
                          <span className="font-medium">{l.meta.taskName}</span>
                        </div>
                        <div className="flex gap-2 items-center">
                          <span className="text-muted-foreground w-14">视频</span>
                          <a
                            href={`${API}/ipfs/${l.meta.videoCID}/`}
                            target="_blank"
                            rel="noreferrer"
                            className="font-mono text-blue-500 hover:underline truncate"
                          >
                            {l.meta.videoCID.slice(0, 18)}...
                          </a>
                        </div>
                        <div className="flex gap-2 items-center">
                          <span className="text-muted-foreground w-14">SFT</span>
                          <span className="font-mono truncate text-muted-foreground">
                            {l.meta.sftCID ? `${l.meta.sftCID.slice(0, 18)}...` : '—'}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Download links (available before purchase) */}
                    {l.meta?.videoCID && (
                      <div className="flex gap-2 pt-1">
                        <a
                          href={`${API}/ipfs/${l.meta.videoCID}/`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs flex items-center gap-1 text-blue-500 hover:underline"
                        >
                          <Download className="w-3 h-3" /> 预览视频
                        </a>
                        {l.meta.sftCID && (
                          <a
                            href={`${API}/ipfs/${l.meta.sftCID}/`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs flex items-center gap-1 text-blue-500 hover:underline"
                          >
                            <Download className="w-3 h-3" /> SFT 标注
                          </a>
                        )}
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t">
                      <span className="font-bold text-base">{l.priceETH} ETH</span>
                      {l.seller.toLowerCase() === account.toLowerCase() ? (
                        <Badge variant="secondary">我的</Badge>
                      ) : (
                        <Button size="sm" disabled={!account} onClick={() => handleBuy(l)}>
                          <ShoppingCart className="w-3 h-3 mr-1" /> 购买
                        </Button>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      卖家: {shortAddr(l.seller)} · 上架时间: {new Date(l.listedAt * 1000).toLocaleDateString('zh-CN')}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* ── Tab: Upload & Mint ── */}
        <TabsContent value="upload" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">上传数据到 IPFS 并铸造 NFT</CardTitle>
              <p className="text-sm text-muted-foreground">
                视频 + SFT 标注包将存入本地 IPFS 模拟存储，元数据写入 Hardhat 链上合约
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>数据集名称 *</Label>
                  <Input
                    value={title}
                    onChange={e => setTitle(e.target.value)}
                    placeholder="screw_fastening_v1"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>机器人类型 *</Label>
                  <Input
                    value={robotType}
                    onChange={e => setRobotType(e.target.value)}
                    placeholder="UR5 / Franka / Unitree H1"
                  />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>任务名称 *</Label>
                <Input
                  value={taskName}
                  onChange={e => setTaskName(e.target.value)}
                  placeholder="screw_tightening"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>视频文件（.mp4 等）</Label>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={e => setVideoFile(e.target.files?.[0] ?? null)}
                    className="text-sm w-full border border-input rounded-md px-3 py-2 bg-background"
                  />
                  {videoFile && (
                    <p className="text-xs text-muted-foreground truncate">{videoFile.name}</p>
                  )}
                </div>
                <div className="space-y-1.5">
                  <Label>SFT 标注包（.jsonl/.zip）</Label>
                  <input
                    type="file"
                    accept=".jsonl,.json,.zip,.tar,.gz,.txt"
                    onChange={e => setSftFile(e.target.files?.[0] ?? null)}
                    className="text-sm w-full border border-input rounded-md px-3 py-2 bg-background"
                  />
                  {sftFile && (
                    <p className="text-xs text-muted-foreground truncate">{sftFile.name}</p>
                  )}
                </div>
              </div>

              {!account && (
                <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-md p-3 text-sm text-amber-800">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  请先点击右上角「连接 MetaMask」再铸造 NFT
                </div>
              )}

              {uploading && (
                <div className="space-y-2">
                  <Progress value={uploadProgress} className="h-2" />
                  <p className="text-sm text-muted-foreground">{uploadStatus}</p>
                </div>
              )}

              {mintResult && (
                <div className="bg-green-50 border border-green-200 rounded-md p-4 space-y-1.5 text-sm">
                  <p className="font-semibold text-green-800 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" /> NFT 铸造成功!
                  </p>
                  <p>Token ID: <span className="font-mono font-bold">{mintResult.tokenId}</span></p>
                  <p className="font-mono text-xs text-muted-foreground">
                    交易哈希: {mintResult.txHash}
                  </p>
                  <p className="font-mono text-xs text-muted-foreground truncate">
                    视频 CID: {mintResult.videoCID || '—'}
                  </p>
                  <p className="font-mono text-xs text-muted-foreground truncate">
                    SFT CID: {mintResult.sftCID || '—'}
                  </p>
                </div>
              )}

              <Button
                className="w-full"
                disabled={uploading || !account || !title || !robotType || !taskName}
                onClick={handleUploadMint}
              >
                <Upload className="w-4 h-4 mr-2" />
                {uploading ? '处理中...' : '上传到 IPFS 并铸造 NFT'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Tab: List for Sale ── */}
        <TabsContent value="sell" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">将我的 NFT 上架到市场</CardTitle>
              <p className="text-sm text-muted-foreground">
                输入你持有的 Token ID 和期望售价，合约将自动处理授权和上架
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Token ID</Label>
                  <Input
                    type="number"
                    min="1"
                    value={listTokenId}
                    onChange={e => setListTokenId(e.target.value)}
                    placeholder="1"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>售价（ETH）</Label>
                  <Input
                    type="number"
                    step="0.001"
                    min="0.001"
                    value={listPrice}
                    onChange={e => setListPrice(e.target.value)}
                    placeholder="0.1"
                  />
                </div>
              </div>

              {txStatus && (
                <div className={`rounded-md p-3 text-sm ${
                  txStatus.startsWith('错误')
                    ? 'bg-red-50 border border-red-200 text-red-700'
                    : 'bg-blue-50 border border-blue-200 text-blue-700'
                }`}>
                  {txStatus}
                </div>
              )}

              <Button
                className="w-full"
                disabled={!account || !listTokenId || !listPrice}
                onClick={handleListForSale}
              >
                <Tag className="w-4 h-4 mr-2" /> 授权并上架出售
              </Button>

              <div className="bg-muted rounded-md p-3 text-xs text-muted-foreground space-y-1">
                <p className="font-medium text-foreground">交易流程说明</p>
                <p>1. MetaMask 签名授权市场合约操作你的 NFT</p>
                <p>2. 调用 listItem() 在链上登记售价</p>
                <p>3. 买家支付 ETH → NFT 自动转移给买家 → 你收到 ETH</p>
                <p>平台手续费: 2.5%</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Setup Guide (shown when chain is not connected) */}
      {!web3Status?.connected && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">本地开发环境启动指南</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs bg-muted rounded-md p-4 font-mono leading-6 overflow-x-auto">{`# 终端 1: 启动 Hardhat 本地以太坊节点
cd Platform/contracts
npm install
npm run node          # 保持运行，显示 20 个测试账户

# 终端 2: 部署智能合约
cd Platform/contracts
npm run deploy:local  # 合约地址自动写入 backend/web3-deployment.json

# 终端 3: 安装后端依赖并启动
cd Platform/backend
npm install           # 安装 ethers 等新依赖
npm run dev

# MetaMask 配置:
#   网络名称: Hardhat Local
#   RPC URL:  http://127.0.0.1:8545
#   Chain ID: 31337
#   导入测试账户私钥 (从终端 1 的 hardhat node 输出复制)`}</pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

"""
存储抽象层（Storage Abstraction Layer）
=========================================
这是整个云化改造的核心设计。

设计目标：让 data_pipeline.py 和 app.py 完全不知道"数据存在哪"。
  - 本地开发：数据存在 market_data/ 文件夹
  - 生产环境：数据存在 Cloudflare R2 / AWS S3

学习目标：
  - 理解"依赖反转原则"（DIP）：高层模块不依赖底层存储细节
  - 掌握 boto3 操作 S3/R2 的核心 API
  - 学习环境变量管理（12-Factor App 的第三条原则）
  - 理解为什么永远不能把密钥写进代码

架构图：
                     ┌─────────────────┐
  data_pipeline.py ──►  get_storage()   ├──► LocalStorage  (本地开发)
  app.py           ──►  [抽象接口]      ├──► S3Storage     (云端生产)
                     └─────────────────┘
                      根据环境变量自动选择

"""

import os
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger("storage")

# ============================================================
# 抽象基类（Storage Interface）
#
# 学习要点 - 为什么要有抽象基类？
#   如果 pipeline 直接调用 boto3，那么：
#     本地测试：必须配置真实的 AWS 凭证（麻烦）
#     单元测试：无法 mock（困难）
#     切换云厂商：要改动所有业务代码（脆弱）
#
#   有了抽象接口，所有调用方只和接口打交道，
#   底层换成 GCS / Azure Blob / 本地文件 只需换一行配置。
#   这就是"面向接口编程"（Program to an Interface）。
#
# 学习要点 - Python ABC（Abstract Base Class）：
#   @abstractmethod 标记的方法必须在子类中实现，
#   否则实例化子类时会报 TypeError。
#   相当于 Java/Go 的 interface 或 TypeScript 的 abstract class。
# ============================================================

class StorageBackend(ABC):
    """存储后端的抽象接口，所有实现必须提供这三个方法。"""

    @abstractmethod
    def read_parquet(self, key: str) -> pd.DataFrame:
        """从存储中读取一个 Parquet 文件，返回 DataFrame。"""
        ...

    @abstractmethod
    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        """将 DataFrame 写入存储中的指定 key。"""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查指定 key 是否存在。"""
        ...


# ============================================================
# 本地文件系统实现
#
# 学习要点 - 本地实现的价值：
#   1. 无需任何云配置，开发时直接用
#   2. 作为"参考实现"，帮助理解接口语义
#   3. 可以作为 CI/CD 中的测试 stub
# ============================================================

class LocalStorage(StorageBackend):
    """
    把 key 映射到本地文件路径。
    key 格式：'market_data/AAPL_historical.parquet'
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)

    def _full_path(self, key: str) -> Path:
        return self.base_dir / key

    def read_parquet(self, key: str) -> pd.DataFrame:
        path = self._full_path(key)
        if not path.exists():
            logger.error(f"[LocalStorage] 文件不存在: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.debug(f"[LocalStorage] 读取: {path} ({len(df)} 行)")
        return df

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine='pyarrow', compression='snappy')
        logger.info(f"[LocalStorage] 写入: {path} ({len(df)} 行)")

    def exists(self, key: str) -> bool:
        return self._full_path(key).exists()


# ============================================================
# S3 / Cloudflare R2 实现
#
# 学习要点 - 为什么推荐 Cloudflare R2 而不是 AWS S3？
#   AWS S3 的收费陷阱：
#     存储费：$0.023/GB/月 ← 便宜
#     PUT 请求费：$0.005/千次 ← 还好
#     下行流量（Egress）费：$0.09/GB ← 这是大坑！
#     你的 Streamlit app 每次读数据都要付出流量费。
#
#   Cloudflare R2：
#     存储费：$0.015/GB/月（比 S3 便宜）
#     PUT/GET 请求费：基本免费（有很大免费额度）
#     下行流量费：$0（完全免费！）← 这是最大的优势
#     API：完全兼容 S3（boto3 稍改 endpoint_url 即可）
#
# 学习要点 - boto3 的核心概念：
#   boto3 是 AWS 的 Python SDK，也是操作 R2 的工具（R2 兼容 S3 API）
#   核心对象：
#     boto3.client('s3', ...)  → 低级 API，更灵活
#     boto3.resource('s3', ...) → 高级 API，面向对象（已不推荐）
#   关键操作：
#     client.get_object(Bucket, Key) → 下载
#     client.put_object(Bucket, Key, Body) → 上传
#     client.head_object(Bucket, Key) → 检查是否存在
#
# 学习要点 - 为什么用 io.BytesIO 做内存中转？
#   Parquet 文件本质是字节流（bytes）。
#   boto3 的 get_object 返回字节流，
#   pd.read_parquet 可以接受文件路径或 BytesIO 对象。
#   用 BytesIO 做中转，避免把临时文件写到磁盘，更高效。
# ============================================================

class S3Storage(StorageBackend):
    """
    Cloudflare R2 / AWS S3 存储实现。

    使用方法（R2）：
      export R2_ACCOUNT_ID=your_account_id
      export R2_ACCESS_KEY_ID=your_access_key
      export R2_SECRET_ACCESS_KEY=your_secret_key
      export R2_BUCKET_NAME=quant-data

    使用方法（AWS S3）：
      export AWS_ACCESS_KEY_ID=...
      export AWS_SECRET_ACCESS_KEY=...
      export S3_BUCKET_NAME=quant-data
      （不需要设置 endpoint_url，boto3 默认连 AWS）
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str = None,   # R2 需要设置；AWS S3 留 None
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        region_name: str = "auto",
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError("请安装 boto3: pip install boto3")

        self.bucket_name = bucket_name

        # 创建 S3 client
        # 学习要点 - endpoint_url 参数的作用：
        #   boto3 默认连接 AWS 的 endpoint（https://s3.amazonaws.com）。
        #   设置 endpoint_url 可以把 boto3 指向任何 S3 兼容的服务：
        #   R2: https://<account_id>.r2.cloudflarestorage.com
        #   MinIO（自建）: http://localhost:9000
        #   这就是"S3 协议"成为业界标准的价值——换供应商只改一个 URL。
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        logger.info(f"[S3Storage] 初始化完成，bucket: {bucket_name}, "
                    f"endpoint: {endpoint_url or 'AWS 默认'}")

    def read_parquet(self, key: str) -> pd.DataFrame:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            # response['Body'] 是一个流对象，read() 读取全部字节
            body_bytes = response['Body'].read()
            # 用 BytesIO 把字节包装成"文件对象"，供 pandas 读取
            df = pd.read_parquet(io.BytesIO(body_bytes))
            logger.info(f"[S3Storage] 读取: s3://{self.bucket_name}/{key} ({len(df)} 行)")
            return df
        except self.client.exceptions.NoSuchKey:
            logger.error(f"[S3Storage] 文件不存在: {key}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"[S3Storage] 读取失败 {key}: {e}")
            return pd.DataFrame()

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        # 先写入内存中的 BytesIO，再上传
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine='pyarrow', compression='snappy')
        buffer.seek(0)  # 重置读取位置到开头（关键！忘记这步会上传空文件）

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream',
        )
        logger.info(f"[S3Storage] 写入: s3://{self.bucket_name}/{key} ({len(df)} 行)")

    def exists(self, key: str) -> bool:
        try:
            # head_object 只获取元数据，不下载文件内容（非常高效）
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False


# ============================================================
# 工厂函数（Factory Function）
#
# 学习要点 - 12-Factor App 的第三条原则：Store config in the environment
#   "好的应用不应该把配置硬编码在代码里，
#    配置应该来自环境变量（Environment Variables）"
#
#   为什么不用配置文件（config.json / .env 文件）？
#     配置文件可能被不小心 commit 进 git → 密钥泄露
#     环境变量是 OS 级别的隔离，不会出现在 git 历史里
#
#   实践规则：
#     ✅ 把 API Key、数据库密码 放在环境变量
#     ✅ 把非敏感的配置（如 batch_size）放在代码里
#     ❌ 永远不要把 Secret 写进代码或提交到 git
#
# 工厂函数的逻辑：
#   如果环境变量里有 R2_BUCKET_NAME → 用 R2
#   如果有 S3_BUCKET_NAME → 用 AWS S3
#   否则 → 用本地文件系统（开发环境默认）
# ============================================================

def get_storage(base_dir: str = ".") -> StorageBackend:
    """
    根据环境变量自动选择并初始化存储后端。

    环境变量配置（在 .env 或 shell 中设置，不要写进代码）：

    Cloudflare R2（推荐）：
      R2_ACCOUNT_ID       = <你的 Cloudflare Account ID>
      R2_ACCESS_KEY_ID    = <R2 API Token 的 Access Key>
      R2_SECRET_ACCESS_KEY= <R2 API Token 的 Secret Key>
      R2_BUCKET_NAME      = quant-data

    AWS S3：
      AWS_ACCESS_KEY_ID     = <IAM 用户的 Access Key>
      AWS_SECRET_ACCESS_KEY = <IAM 用户的 Secret Key>
      S3_BUCKET_NAME        = quant-data

    本地（开发环境，无需配置任何变量）：
      不设置任何变量 → 自动使用本地文件系统
    """
    # 优先检查 R2
    r2_bucket = os.getenv("R2_BUCKET_NAME")
    r2_account_id = os.getenv("R2_ACCOUNT_ID")

    if r2_bucket and r2_account_id:
        logger.info("检测到 R2 配置，使用 Cloudflare R2 存储")
        return S3Storage(
            bucket_name=r2_bucket,
            endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

    # 其次检查 AWS S3
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if s3_bucket:
        logger.info("检测到 S3 配置，使用 AWS S3 存储")
        return S3Storage(
            bucket_name=s3_bucket,
            endpoint_url=None,  # 使用 AWS 默认 endpoint
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    # 默认：本地文件系统
    logger.info("未检测到云存储配置，使用本地文件系统（开发模式）")
    return LocalStorage(base_dir=base_dir)

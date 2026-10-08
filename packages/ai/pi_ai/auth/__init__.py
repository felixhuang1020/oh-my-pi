"""认证基础设施：credential、provider 认证方式与认证解析。

本包是 ``packages/ai/src/auth/`` 的 Python 移植，对外暴露 credential 类型、
:class:`~pi_ai.auth.types.CredentialStore` 协议、默认的内存存储与认证解析入口。
本项目只支持 api-key 认证。
"""

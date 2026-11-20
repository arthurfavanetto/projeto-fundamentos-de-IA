# ==============================================================================
# ChromaDB 1.5.9 — Hardened Non-Root Wrapper (UID/GID 10001)
# ==============================================================================
# Baseado integralmente na imagem oficial pinada chromadb/chroma:1.5.9.
# Preserva ENTRYPOINT ["dumb-init", "--", "chroma"] e CMD ["run", "/config.yaml"]
# originais da distribuição oficial, adicionando apenas execução sem privilégios
# de root e permissão de leitura/escrita em /data.

FROM chromadb/chroma:1.5.9

RUN groupadd --gid 10001 chroma \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin chroma \
    && mkdir -p /data \
    && chown -R 10001:10001 /data \
    && chmod 0750 /data \
    && chmod 0644 /config.yaml

USER 10001:10001

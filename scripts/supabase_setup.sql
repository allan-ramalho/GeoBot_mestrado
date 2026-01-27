-- Supabase SQL Setup Script
-- Execute este script no SQL Editor do Supabase

-- Habilitar extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Criar tabela de documentos com embeddings
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024),  -- E5-Large dimension
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Criar índice para busca vetorial
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Função RPC para busca de similaridade
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_count INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Políticas de segurança (ajuste conforme necessário)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Permitir leitura para todos (ajuste conforme sua necessidade)
CREATE POLICY "Allow read access to all users"
    ON documents
    FOR SELECT
    USING (true);

-- Permitir insert apenas para authenticated users
CREATE POLICY "Allow insert for authenticated users"
    ON documents
    FOR INSERT
    WITH CHECK (auth.role() = 'authenticated');

-- Comentários
COMMENT ON TABLE documents IS 'Armazena chunks de documentos com embeddings para RAG';
COMMENT ON COLUMN documents.embedding IS 'Vetor de embedding gerado por E5-Large (1024 dimensões)';
COMMENT ON COLUMN documents.metadata IS 'Metadata do documento: título, autores, ano, fonte, etc.';
COMMENT ON FUNCTION match_documents IS 'Busca de similaridade vetorial para RAG';

-- =====================================================
-- SnipShot Database Migration - Supabase PostgreSQL
-- =====================================================
-- Run this in Supabase SQL Editor to set up the database
-- 
-- This creates/updates:
-- 1. folders table - for organizing translated images
-- 2. images table - metadata for translated images
-- =====================================================

-- =====================================================
-- FOLDERS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS folders (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for faster user queries
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);

-- Unique constraint: user can't have duplicate folder names
CREATE UNIQUE INDEX IF NOT EXISTS idx_folders_user_name 
    ON folders(user_id, name);

-- =====================================================
-- IMAGES TABLE - Create if not exists
-- =====================================================
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    storage_path TEXT NOT NULL,
    public_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- IMAGES TABLE - Add missing columns (for existing tables)
-- =====================================================
-- Add folder_id column
ALTER TABLE images ADD COLUMN IF NOT EXISTS folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL;

-- Add filename column
ALTER TABLE images ADD COLUMN IF NOT EXISTS filename VARCHAR(255);

-- Add original_filename column
ALTER TABLE images ADD COLUMN IF NOT EXISTS original_filename VARCHAR(255);

-- Add source_language column
ALTER TABLE images ADD COLUMN IF NOT EXISTS source_language VARCHAR(10);

-- Add target_language column
ALTER TABLE images ADD COLUMN IF NOT EXISTS target_language VARCHAR(10);

-- Add file_size column
ALTER TABLE images ADD COLUMN IF NOT EXISTS file_size INTEGER;

-- Add updated_at column
ALTER TABLE images ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

-- Update existing rows: set filename from original_filename if null
UPDATE images SET filename = COALESCE(original_filename, 'untitled.png') WHERE filename IS NULL;

-- Now make filename NOT NULL (after populating existing rows)
ALTER TABLE images ALTER COLUMN filename SET NOT NULL;

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);
CREATE INDEX IF NOT EXISTS idx_images_folder_id ON images(folder_id);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at DESC);

-- =====================================================
-- SECURITY NOTE
-- =====================================================
-- RLS is DISABLED because:
-- 1. We use SQLAlchemy/asyncpg for direct PostgreSQL access
-- 2. auth.uid() only works with Supabase client, not direct connections
-- 3. Our API verifies JWT tokens and filters by user_id in all queries
-- 
-- Security is handled at the API layer:
-- - JWT verification on every request
-- - All queries filter by authenticated user_id
-- - Users can only access their own data
-- =====================================================

-- Disable RLS on folders
ALTER TABLE folders DISABLE ROW LEVEL SECURITY;

-- Disable RLS on images  
ALTER TABLE images DISABLE ROW LEVEL SECURITY;

-- Drop any existing policies (cleanup)
DROP POLICY IF EXISTS "Users can view own folders" ON folders;
DROP POLICY IF EXISTS "Users can insert own folders" ON folders;
DROP POLICY IF EXISTS "Users can update own folders" ON folders;
DROP POLICY IF EXISTS "Users can delete own folders" ON folders;

DROP POLICY IF EXISTS "Users can view own images" ON images;
DROP POLICY IF EXISTS "Users can insert own images" ON images;
DROP POLICY IF EXISTS "Users can update own images" ON images;
DROP POLICY IF EXISTS "Users can delete own images" ON images;

-- =====================================================
-- VERIFY TABLES
-- =====================================================
-- Run this to verify the tables were created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
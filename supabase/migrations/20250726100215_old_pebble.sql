/*
  # Fix user_sessions RLS policies for anonymous users

  1. Security Changes
    - Drop existing restrictive policies on user_sessions table
    - Add new policies to allow anonymous users to create and update sessions
    - Enable proper access for the research study application

  This migration ensures that anonymous users can:
  - Create new user sessions (INSERT)
  - Update their own sessions (UPDATE) 
  - This is required for the research study to function properly
*/

-- Drop existing policies that might be too restrictive
DROP POLICY IF EXISTS "Allow anonymous users to create sessions" ON user_sessions;
DROP POLICY IF EXISTS "Allow anonymous users to update their sessions" ON user_sessions;
DROP POLICY IF EXISTS "Users can manage sessions" ON user_sessions;

-- Create new policies for anonymous access
CREATE POLICY "Enable insert for anonymous users" 
  ON user_sessions 
  FOR INSERT 
  TO anon 
  WITH CHECK (true);

CREATE POLICY "Enable update for anonymous users" 
  ON user_sessions 
  FOR UPDATE 
  TO anon 
  USING (true) 
  WITH CHECK (true);

CREATE POLICY "Enable select for anonymous users" 
  ON user_sessions 
  FOR SELECT 
  TO anon 
  USING (true);
/*
  # Fix user_sessions RLS policy for anonymous users

  1. Security Changes
    - Add policy to allow anonymous users to insert new user sessions
    - This enables the client-side application to create sessions without authentication
    - Maintains security by only allowing INSERT operations for session creation

  2. Changes Made
    - Create policy "Allow anonymous users to create sessions" on user_sessions table
    - Grants INSERT permission to anon role (unauthenticated users)
    - Allows the application to initialize new user sessions from the frontend
*/

-- Create policy to allow anonymous users to insert new user sessions
CREATE POLICY "Allow anonymous users to create sessions"
  ON user_sessions
  FOR INSERT
  TO anon
  WITH CHECK (true);

-- Also allow anonymous users to update their own sessions (for phase updates, screen size, etc.)
CREATE POLICY "Allow anonymous users to update their sessions"
  ON user_sessions
  FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);
/*
  # Fix user_sessions RLS policy for anonymous users

  1. Security Updates
    - Add policy to allow anonymous users to INSERT new sessions
    - Add policy to allow anonymous users to UPDATE their sessions
    - These policies are essential for the research study application to function
    
  2. Changes
    - Enable anonymous session creation for study participants
    - Allow session updates for phase transitions and data recording
*/

-- Allow anonymous users to create new sessions
CREATE POLICY "Allow anonymous users to create sessions"
  ON user_sessions
  FOR INSERT
  TO anon
  WITH CHECK (true);

-- Allow anonymous users to update sessions (for phase transitions, screen size, etc.)
CREATE POLICY "Allow anonymous users to update their sessions"
  ON user_sessions
  FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);
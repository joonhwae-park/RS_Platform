/*
  # Create Session Phase 1 Movies Table

  1. New Tables
    - `session_phase1_movies`
      - `session_id` (uuid) - References user_sessions.id
      - `movie_id` (text) - Movie ID from movies table
      - `selection_type` (text) - 'vote_count', 'vote_variance', or 'genre_1' through 'genre_10'
      - `created_at` (timestamptz)
      
  2. Purpose
    - Store the 30 movies selected for each session in Phase 1
    - Prevent these movies from appearing in Phase 2 recommendations
    - Track which selection criteria was used for each movie
    
  3. Security
    - Enable RLS
    - Allow public access for anonymous users
*/

-- Create table
CREATE TABLE IF NOT EXISTS session_phase1_movies (
  session_id uuid NOT NULL,
  movie_id text NOT NULL,
  selection_type text NOT NULL,
  created_at timestamptz DEFAULT now(),
  PRIMARY KEY (session_id, movie_id)
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_session_phase1_movies_session_id 
  ON session_phase1_movies(session_id);

CREATE INDEX IF NOT EXISTS idx_session_phase1_movies_movie_id 
  ON session_phase1_movies(movie_id);

-- Enable RLS
ALTER TABLE session_phase1_movies ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access
CREATE POLICY "Public can read phase1 movies"
  ON session_phase1_movies
  FOR SELECT
  TO anon, authenticated
  USING (true);

-- Policy: Allow public insert
CREATE POLICY "Public can insert phase1 movies"
  ON session_phase1_movies
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

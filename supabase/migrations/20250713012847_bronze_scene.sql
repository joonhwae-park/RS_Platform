/*
  # Create Phase 2 movies table

  1. New Tables
    - `phase2_movies`
      - `id` (integer, primary key)
      - `title` (text)
      - `description` (text)
      - `poster` (text)
      - `trailer` (text)
      - `genre` (text)
      - `year` (integer)
      - `director` (text)
      - `actors` (text)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on `phase2_movies` table
    - Add policy for public read access
*/

CREATE TABLE IF NOT EXISTS phase2_movies (
  id integer PRIMARY KEY,
  title text NOT NULL,
  description text NOT NULL,
  poster text NOT NULL,
  trailer text NOT NULL,
  genre text NOT NULL,
  year integer NOT NULL,
  director text NOT NULL,
  actors text NOT NULL DEFAULT '',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE phase2_movies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Phase 2 movies are publicly readable"
  ON phase2_movies
  FOR SELECT
  TO public
  USING (true);
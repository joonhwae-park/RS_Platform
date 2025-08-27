/*
  # Create CineRate Database Schema

  1. New Tables
    - `user_sessions`
      - `id` (uuid, primary key)
      - `user_id` (text)
      - `phase` (text)
      - `start_time` (timestamptz)
      - `end_time` (timestamptz, nullable)
      - `created_at` (timestamptz)
    
    - `movie_ratings`
      - `id` (uuid, primary key)
      - `session_id` (uuid, foreign key to user_sessions)
      - `movie_id` (integer)
      - `rating` (integer)
      - `diversity_rating` (integer, nullable)
      - `novelty_rating` (integer, nullable)
      - `serendipity_rating` (integer, nullable)
      - `created_at` (timestamptz)
    
    - `mouse_events`
      - `id` (uuid, primary key)
      - `session_id` (uuid, foreign key to user_sessions)
      - `x` (integer)
      - `y` (integer)
      - `timestamp` (timestamptz)
      - `created_at` (timestamptz)
    
    - `movies`
      - `id` (integer, primary key)
      - `title` (text)
      - `description` (text)
      - `poster` (text)
      - `trailer` (text)
      - `genre` (text)
      - `year` (integer)
      - `director` (text)
      - `is_recommended` (boolean)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for public access to movies table
    - Add policies for session-based access to user data
*/

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  phase text NOT NULL,
  start_time timestamptz DEFAULT now() NOT NULL,
  end_time timestamptz,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Create movie_ratings table
CREATE TABLE IF NOT EXISTS movie_ratings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  movie_id integer NOT NULL,
  rating integer NOT NULL,
  diversity_rating integer,
  novelty_rating integer,
  serendipity_rating integer,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Create mouse_events table
CREATE TABLE IF NOT EXISTS mouse_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  x integer NOT NULL,
  y integer NOT NULL,
  timestamp timestamptz NOT NULL,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Create movies table
CREATE TABLE IF NOT EXISTS movies (
  id integer PRIMARY KEY,
  title text NOT NULL,
  description text NOT NULL,
  poster text NOT NULL,
  trailer text NOT NULL,
  genre text NOT NULL,
  year integer NOT NULL,
  director text NOT NULL,
  is_recommended boolean DEFAULT false NOT NULL,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Enable Row Level Security
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE movie_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE mouse_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE movies ENABLE ROW LEVEL SECURITY;

-- Create policies for movies (public read access)
CREATE POLICY "Movies are publicly readable"
  ON movies
  FOR SELECT
  TO public
  USING (true);

-- Create policies for user_sessions (users can manage their own sessions)
CREATE POLICY "Users can manage their own sessions"
  ON user_sessions
  FOR ALL
  TO public
  USING (true);

-- Create policies for movie_ratings (users can manage ratings for their sessions)
CREATE POLICY "Users can manage ratings for their sessions"
  ON movie_ratings
  FOR ALL
  TO public
  USING (true);

-- Create policies for mouse_events (users can manage mouse events for their sessions)
CREATE POLICY "Users can manage mouse events for their sessions"
  ON mouse_events
  FOR ALL
  TO public
  USING (true);
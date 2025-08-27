/*
  # Add actors and minimum requirements

  1. Schema Changes
    - Add `actors` column to movies table
    - Add `minimum_ratings_required` column to user_sessions table
  
  2. Data Updates
    - Update existing movies with actor information
    - Add 10 more initial movies (21-30) to reach 30 total
*/

-- Add actors column to movies table
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'movies' AND column_name = 'actors'
  ) THEN
    ALTER TABLE movies ADD COLUMN actors text NOT NULL DEFAULT '';
  END IF;
END $$;

-- Add minimum_ratings_required column to user_sessions table
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'user_sessions' AND column_name = 'minimum_ratings_required'
  ) THEN
    ALTER TABLE user_sessions ADD COLUMN minimum_ratings_required integer DEFAULT 5;
  END IF;
END $$;

-- Update existing movies with actor information
UPDATE movies SET actors = 'Tim Robbins, Morgan Freeman, Bob Gunton' WHERE id = 1;
UPDATE movies SET actors = 'Marlon Brando, Al Pacino, James Caan' WHERE id = 2;
UPDATE movies SET actors = 'Christian Bale, Heath Ledger, Aaron Eckhart' WHERE id = 3;
UPDATE movies SET actors = 'John Travolta, Uma Thurman, Samuel L. Jackson' WHERE id = 4;
UPDATE movies SET actors = 'Tom Hanks, Robin Wright, Gary Sinise' WHERE id = 5;
UPDATE movies SET actors = 'Leonardo DiCaprio, Marion Cotillard, Tom Hardy' WHERE id = 6;
UPDATE movies SET actors = 'Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss' WHERE id = 7;
UPDATE movies SET actors = 'Robert De Niro, Ray Liotta, Joe Pesci' WHERE id = 8;
UPDATE movies SET actors = 'Elijah Wood, Ian McKellen, Orlando Bloom' WHERE id = 9;
UPDATE movies SET actors = 'Brad Pitt, Edward Norton, Helena Bonham Carter' WHERE id = 10;
UPDATE movies SET actors = 'Matthew McConaughey, Anne Hathaway, Jessica Chastain' WHERE id = 11;
UPDATE movies SET actors = 'Leonardo DiCaprio, Matt Damon, Jack Nicholson' WHERE id = 12;
UPDATE movies SET actors = 'Jamie Foxx, Christoph Waltz, Leonardo DiCaprio' WHERE id = 13;
UPDATE movies SET actors = 'Hugh Jackman, Christian Bale, Michael Caine' WHERE id = 14;
UPDATE movies SET actors = 'Miles Teller, J.K. Simmons, Paul Reiser' WHERE id = 15;
UPDATE movies SET actors = 'Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong' WHERE id = 16;
UPDATE movies SET actors = 'Joaquin Phoenix, Robert De Niro, Zazie Beetz' WHERE id = 17;
UPDATE movies SET actors = 'George MacKay, Dean-Charles Chapman, Mark Strong' WHERE id = 18;
UPDATE movies SET actors = 'Tom Hardy, Charlize Theron, Nicholas Hoult' WHERE id = 19;
UPDATE movies SET actors = 'Ryan Gosling, Harrison Ford, Ana de Armas' WHERE id = 20;

-- Update recommended movies with actor information
UPDATE movies SET actors = 'Timothée Chalamet, Rebecca Ferguson, Oscar Isaac' WHERE id = 21;
UPDATE movies SET actors = 'Michelle Yeoh, Stephanie Hsu, Ke Huy Quan' WHERE id = 22;
UPDATE movies SET actors = 'Ralph Fiennes, F. Murray Abraham, Mathieu Amalric' WHERE id = 23;
UPDATE movies SET actors = 'Trevante Rhodes, André Holland, Janelle Monáe' WHERE id = 24;
UPDATE movies SET actors = 'Joaquin Phoenix, Scarlett Johansson, Amy Adams' WHERE id = 25;
UPDATE movies SET actors = 'Amy Adams, Jeremy Renner, Forest Whitaker' WHERE id = 26;
UPDATE movies SET actors = 'Ryan Gosling, Emma Stone, John Legend' WHERE id = 27;
UPDATE movies SET actors = 'Sally Hawkins, Michael Shannon, Richard Jenkins' WHERE id = 28;
UPDATE movies SET actors = 'Daniel Craig, Chris Evans, Ana de Armas' WHERE id = 29;
UPDATE movies SET actors = 'Robert Pattinson, Willem Dafoe' WHERE id = 30;

-- Move recommended movies to initial phase and add 10 new movies for initial rating
UPDATE movies SET is_recommended = false WHERE id BETWEEN 21 AND 30;

-- Insert 10 new recommended movies (id 31-40)
INSERT INTO movies (id, title, description, poster, trailer, genre, year, director, actors, is_recommended) VALUES
(31, 'Nomadland', 'A woman in her sixties embarks on a journey through the western United States after losing everything in the Great Recession.', 'https://images.pexels.com/photos/8263502/pexels-photo-8263502.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/6sxCFZ8_d84', 'Drama', 2020, 'Chloé Zhao', 'Frances McDormand, David Strathairn, Linda May', true),
(32, 'Minari', 'A Korean family starts a farm in 1980s Arkansas.', 'https://images.pexels.com/photos/8263503/pexels-photo-8263503.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/KQ0gFidlro8', 'Drama', 2020, 'Lee Isaac Chung', 'Steven Yeun, Yeri Han, Alan Kim', true),
(33, 'Sound of Metal', 'A heavy-metal drummer begins to lose his hearing and has to come to grips with a future that will be filled with silence.', 'https://images.pexels.com/photos/8263504/pexels-photo-8263504.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/VFOrGkAvjAE', 'Drama', 2019, 'Darius Marder', 'Riz Ahmed, Olivia Cooke, Paul Raci', true),
(34, 'The Father', 'A man refuses assistance from his daughter as he ages. As he tries to make sense of his changing circumstances, he begins to doubt his loved ones.', 'https://images.pexels.com/photos/8263505/pexels-photo-8263505.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/NkHF_SIqqJ0', 'Drama', 2020, 'Florian Zeller', 'Anthony Hopkins, Olivia Colman, Mark Gatiss', true),
(35, 'Promising Young Woman', 'A young woman, traumatized by a tragic event in her past, seeks out vengeance against those who crossed her path.', 'https://images.pexels.com/photos/8263506/pexels-photo-8263506.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/7i5kiFDunk8', 'Thriller', 2020, 'Emerald Fennell', 'Carey Mulligan, Bo Burnham, Alison Brie', true),
(36, 'Mank', 'The story of Herman J. Mankiewicz and his development of the screenplay for Citizen Kane.', 'https://images.pexels.com/photos/8263507/pexels-photo-8263507.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/aSfBz4_QC0g', 'Biography', 2020, 'David Fincher', 'Gary Oldman, Amanda Seyfried, Lily Collins', true),
(37, 'Soul', 'A musician who has lost his passion for music is transported out of his body and must find his way back with the help of an infant soul learning about herself.', 'https://images.pexels.com/photos/8263508/pexels-photo-8263508.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/xOsLIiBStEs', 'Animation', 2020, 'Pete Docter', 'Jamie Foxx, Tina Fey, Graham Norton', true),
(38, 'Another Round', 'Four friends, all high school teachers, test a theory that they will improve their lives by maintaining a constant level of alcohol in their blood.', 'https://images.pexels.com/photos/8263509/pexels-photo-8263509.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/PlWKhbxWf4g', 'Drama', 2020, 'Thomas Vinterberg', 'Mads Mikkelsen, Thomas Bo Larsen, Magnus Millang', true),
(39, 'Judas and the Black Messiah', 'The story of Fred Hampton, Chairman of the Illinois Black Panther Party, and his fateful betrayal by FBI informant William O''Neal.', 'https://images.pexels.com/photos/8263510/pexels-photo-8263510.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/sSjtGqRXQ9Y', 'Biography', 2021, 'Shaka King', 'Daniel Kaluuya, LaKeith Stanfield, Jesse Plemons', true),
(40, 'The Trial of the Chicago 7', 'The story of 7 people on trial stemming from various charges surrounding the uprising at the 1968 Democratic National Convention.', 'https://images.pexels.com/photos/8263511/pexels-photo-8263511.jpeg?auto=compress&cs=tinysrgb&w=400', 'https://www.youtube.com/embed/PdxgBYbkJAg', 'Drama', 2020, 'Aaron Sorkin', 'Eddie Redmayne, Alex Sharp, Sacha Baron Cohen', true);
import React, { useState } from 'react';
import { Movie } from '../types';
import { SelectRating } from './SelectRating';
import { TrailerModal } from './TrailerModal';
import { Play, Calendar, User, Users, ChevronDown, ChevronUp, Info, StickyNote, FileText } from 'lucide-react';

interface MovieCardProps {
  movie: Movie;
  rating: number;
  onRatingChange: (rating: number) => void;
  showAdditionalQuestions?: boolean;
  diversityRating?: number;
  noveltyRating?: number;
  serendipityRating?: number;
  onDiversityChange?: (rating: number) => void;
  onNoveltyChange?: (rating: number) => void;
  onSerendipityChange?: (rating: number) => void;
  onTrailerStart?: (movieId: string) => void;
  onTrailerEnd?: (movieId: string) => void;
}

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  rating,
  onRatingChange,
  showAdditionalQuestions = false,
  diversityRating = -1,
  noveltyRating = -1,
  serendipityRating = -1,
  onDiversityChange,
  onNoveltyChange,
  onSerendipityChange,
  onTrailerStart,
  onTrailerEnd
}) => {
  const [isDescriptionExpanded, setIsDescriptionExpanded] = useState(false);
  const [isTrailerModalOpen, setIsTrailerModalOpen] = useState(false);
  
  const handleTrailerClick = () => {
    if (onTrailerStart) {
      onTrailerStart(movie.id);
    }
    setIsTrailerModalOpen(true);
  };

  const handleTrailerClose = () => {
    if (onTrailerEnd) {
      onTrailerEnd(movie.id);
    }
    setIsTrailerModalOpen(false);
  };

  const toggleDescription = () => {
    setIsDescriptionExpanded(!isDescriptionExpanded);
  };

  // Determine if description needs truncation (more than 160 characters)
  const shouldTruncate = movie.description.length > 160;
  const truncatedDescription = shouldTruncate 
    ? movie.description.substring(0, 160) + '...'
    : movie.description;

  return (
    <>
      <div className="bg-gray-900 rounded-xl overflow-hidden shadow-2xl hover:shadow-3xl transition-all duration-300 hover:scale-105 border border-gray-800 flex flex-col h-full">
        <div className="relative group">
          <img
            src={movie.poster}
            alt={movie.title}
            className="w-full h-80 object-contain bg-gray-800"
          />
          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-all duration-300 flex items-center justify-center">
            <button
              onClick={handleTrailerClick}
              className="opacity-0 group-hover:opacity-100 bg-amber-500 hover:bg-amber-600 text-black p-4 rounded-full transition-all duration-300 transform scale-0 group-hover:scale-100"
            >
              <Play size={24} />
            </button>
          </div>
        </div>
        
        <div className="p-6 flex flex-col flex-grow">
          <div className="flex items-center gap-4 mb-3 text-sm text-gray-400">
            <div className="flex items-center gap-1">
              <Calendar size={16} />
              <span>{movie.year}</span>
            </div>
            <div className="flex items-center gap-1">
              <User size={16} />
              <span>{movie.director}</span>
            </div>
            <span className="bg-amber-500 text-black px-3 py-1 rounded text-xs font-medium">
              {movie.genre}
            </span>
          </div>
          
          <h3 className="text-xl font-bold text-white mb-3">{movie.title}</h3>
          
          <div className="mb-4 flex-shrink-0">
            <div className="flex items-center gap-1 mb-2">
              <Users size={16} className="text-amber-400" />
              <span className="text-amber-400 font-medium text-sm">Cast</span>
            </div>
            <p className="text-gray-300 text-sm">{movie.actors}</p>
          </div>
          
          <div className="mb-6 flex-grow">
            <div className="flex items-center gap-1 mb-2">
              <FileText size={14} className="text-white" />
              <span className="text-white font-medium text-sm">Description</span>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">
              {isDescriptionExpanded ? movie.description : truncatedDescription}
            </p>
            
            {shouldTruncate && (
              <button
                onClick={toggleDescription}
                className="flex items-center gap-1 mt-2 text-amber-400 hover:text-amber-300 text-sm font-medium transition-colors duration-200"
              >
                {isDescriptionExpanded ? (
                  <>
                    <span>See Less</span>
                    <ChevronUp size={16} />
                  </>
                ) : (
                  <>
                    <span>See More</span>
                    <ChevronDown size={16} />
                  </>
                )}
              </button>
            )}
          </div>
          
          <div className="space-y-4 flex-shrink-0">
            <SelectRating
              rating={rating}
              onRatingChange={onRatingChange}
            label={showAdditionalQuestions ? "* Your Rating for This Movie" : "Your Rating"}
            />
            
            {showAdditionalQuestions && (
              <>
                <div className="mb-3 p-3 bg-gray-800 border border-gray-700 rounded-lg">
                  <p className="text-amber-400 text-sm flex items-center gap-2">
                    <Info size={18} className="text-amber-400 flex-shrink-0" />
                    <span>Hover over the <Info size={16} className="inline-block text-gray-400 mx-1" /> icons to see definitions</span>
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-amber-400 font-medium text-sm">* How diverse is this movie recommendation? (0 = not diverse, 10 = very diverse)</span>
                      <div className="relative group">
                        <Info size={20} className="text-gray-400 hover:text-amber-400 cursor-help" />
                        <div className="absolute bottom-full right-0 mb-2 px-3 py-2 bg-gray-800 border border-gray-600 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none w-64 z-50 shadow-xl">
                          How different is this recommended movie from other movies in this list?
                        </div>
                      </div>
                    </div>
                    <SelectRating
                      rating={diversityRating}
                      onRatingChange={onDiversityChange!}
                      label=""
                    />
                  </div>
                  
                  <div className="relative">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-amber-400 font-medium text-sm">* How novel is this movie recommendation? (0 = not novel, 10 = very novel)</span>
                      <div className="relative group">
                        <Info size={20} className="text-gray-400 hover:text-amber-400 cursor-help" />
                        <div className="absolute bottom-full right-0 mb-2 px-3 py-2 bg-gray-800 border border-gray-600 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none w-64 z-50 shadow-xl">
                          How new, fresh, or unfamiliar is this movie to you?
                        </div>
                      </div>
                    </div>
                    <SelectRating
                      rating={noveltyRating}
                      onRatingChange={onNoveltyChange!}
                      label=""
                    />
                  </div>
                  
                  <div className="relative">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-amber-400 font-medium text-sm">* How serendipitous is this movie recommendation? (0 = not serendipitous, 10 = very serendipitous)</span>
                      <div className="relative group">
                        <Info size={20} className="text-gray-400 hover:text-amber-400 cursor-help" />
                        <div className="absolute bottom-full right-0 mb-2 px-3 py-2 bg-gray-800 border border-gray-600 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none w-64 z-50 shadow-xl">
                          How surprising or unexpected is this recommended movie for you?
                        </div>
                      </div>
                    </div>
                    <SelectRating
                      rating={serendipityRating}
                      onRatingChange={onSerendipityChange!}
                      label=""
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <TrailerModal
        isOpen={isTrailerModalOpen}
        onClose={handleTrailerClose}
        trailerUrl={movie.trailer}
        movieTitle={movie.title}
      />
    </>
  );
};
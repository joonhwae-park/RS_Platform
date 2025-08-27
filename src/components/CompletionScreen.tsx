import React from 'react';
import { Trophy, Star, BarChart3, Heart } from 'lucide-react';

interface CompletionScreenProps {
  totalRatings: number;
  averageRating: number;
}

export const CompletionScreen: React.FC<CompletionScreenProps> = ({
  totalRatings,
  averageRating
}) => {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="max-w-2xl mx-auto text-center">
        <div className="mb-8">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-amber-500 rounded-full mb-6">
            <Trophy size={48} className="text-black" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">
            Congratulations!
          </h1>
          <p className="text-xl text-gray-300 mb-8">
            You've successfully completed the movie rating study!
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex items-center justify-center mb-4">
              <Star className="text-amber-400 mr-2" size={24} />
              <span className="text-amber-400 font-semibold">Total Ratings</span>
            </div>
            <div className="text-3xl font-bold text-white">{totalRatings}</div>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex items-center justify-center mb-4">
              <BarChart3 className="text-amber-400 mr-2" size={24} />
              <span className="text-amber-400 font-semibold">Average Rating</span>
            </div>
            <div className="text-3xl font-bold text-white">{averageRating.toFixed(1)}/10</div>
          </div>
        </div>

        <div className="bg-amber-500 bg-opacity-10 border border-amber-500 rounded-lg p-8">
          <div className="flex items-center justify-center mb-4">
            <Heart className="text-amber-400 mr-2" size={24} />
            <span className="text-amber-400 font-semibold text-lg">Thank You!</span>
          </div>
          <p className="text-white text-lg text-center leading-relaxed">
            Thank you for participating in our study! You can now close the window.
          </p>
        </div>
      </div>
    </div>
  );
};
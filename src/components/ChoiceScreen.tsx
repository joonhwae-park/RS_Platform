import React from 'react';
import { Lightbulb, Star } from 'lucide-react';

interface ChoiceScreenProps {
  ratingsCount: number;
  onGetRecommendations: () => void;
  onRateMore: () => void;
}

export const ChoiceScreen: React.FC<ChoiceScreenProps> = ({
  ratingsCount,
  onGetRecommendations,
  onRateMore
}) => {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="max-w-2xl mx-auto text-center">
        <div className="mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-amber-500 rounded-full mb-6">
            <Star size={40} className="text-black" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">
            Great Progress!
          </h1>
          <p className="text-xl text-gray-300 mb-8">
            You've rated {ratingsCount} movies. What would you like to do next?
          </p>
        </div>

        <div className="bg-amber-500 bg-opacity-10 border border-amber-500 rounded-lg p-6 mb-8">
          <div className="flex items-center justify-center mb-4">
            <Lightbulb className="text-amber-400 mr-2" size={24} />
            <span className="text-amber-400 font-semibold">Pro Tip</span>
          </div>
          <p className="text-white text-lg">
            The more movies you rate, the more accurate your recommendations will be!
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <button
            onClick={onGetRecommendations}
            className="bg-amber-500 hover:bg-amber-600 text-black font-semibold py-4 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
          >
            <div className="text-xl mb-2">Get Recommendations</div>
            <div className="text-sm opacity-90">See movies picked just for you</div>
          </button>
          
          <button
            onClick={onRateMore}
            className="bg-gray-800 hover:bg-gray-700 text-white font-semibold py-4 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl border border-gray-600"
          >
            <div className="text-xl mb-2">Rate More Movies</div>
            <div className="text-sm opacity-70">Improve recommendation accuracy</div>
          </button>
        </div>
      </div>
    </div>
  );
};
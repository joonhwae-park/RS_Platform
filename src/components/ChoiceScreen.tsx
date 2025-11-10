import React from 'react';
import { Lightbulb, Star, Loader2, Info } from 'lucide-react';

interface ChoiceScreenProps {
  ratingsCount: number;
  onGetRecommendations: () => void;
  onRateMore: () => void;
  isGeneratingRecommendations?: boolean;
}

export const ChoiceScreen: React.FC<ChoiceScreenProps> = ({
  ratingsCount,
  onGetRecommendations,
  onRateMore,
  isGeneratingRecommendations = false
}) => {
  if (isGeneratingRecommendations) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-8">
            <Loader2 className="animate-spin text-amber-500 mx-auto mb-6" size={64} />
            <h2 className="text-3xl font-bold text-white mb-4">
              Generating Your Personalized Recommendations
            </h2>
            <p className="text-xl text-gray-400 mb-2">
              Analyzing your ratings and preferences...
            </p>
            <p className="text-lg text-gray-500">
              This may take a few moments
            </p>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 space-y-6">
            <div className="mb-4 p-3 bg-amber-900 bg-opacity-20 border border-amber-600 rounded-lg">
              <p className="text-amber-400 text-sm flex items-center gap-2">
                <Info size={18} className="text-amber-400 flex-shrink-0" />
                <span>When recommendations appear, you can hover over the <Info size={16} className="inline-block text-gray-400 mx-1" /> icons to see definitions</span>
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-amber-400 mb-3">Understanding Recommendation Qualities</h3>

              <div className="space-y-4">
                <div className="bg-gray-900 p-4 rounded-lg">
                  <h4 className="text-lg font-semibold text-white mb-2">Diversity</h4>
                  <p className="text-gray-300">
                    How different is this recommended movie from other movies in the list?
                  </p>
                </div>

                <div className="bg-gray-900 p-4 rounded-lg">
                  <h4 className="text-lg font-semibold text-white mb-2">Novelty</h4>
                  <p className="text-gray-300">
                    How new, fresh, or unfamiliar is this movie to you?
                  </p>
                </div>

                <div className="bg-gray-900 p-4 rounded-lg">
                  <h4 className="text-lg font-semibold text-white mb-2">Serendipity</h4>
                  <p className="text-gray-300">
                    How surprising or unexpected is this recommended movie for you?
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
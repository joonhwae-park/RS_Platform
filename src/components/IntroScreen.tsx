import React from 'react';
import { Film, Star, Target, Award, ArrowRight } from 'lucide-react';

interface IntroScreenProps {
  onStart: () => void;
}

export const IntroScreen: React.FC<IntroScreenProps> = ({ onStart }) => {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="max-w-4xl mx-auto text-center">
        <div className="mb-8">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-amber-500 rounded-full mb-6">
            <Film size={48} className="text-black" />
          </div>
          <h1 className="text-5xl font-bold text-white mb-4">
            Welcome to CineRate
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Please follow the steps below to help us understand your movie preferences. 
          </p>
          
          <div className="bg-amber-500 bg-opacity-10 border border-amber-500 rounded-xl p-6 mb-8 max-w-3xl mx-auto">
            <h3 className="text-amber-400 font-semibold text-lg mb-3">Important Notice</h3>
            <p className="text-gray-300 leading-relaxed">
              Providing more frequent and accurate ratings will lead to more precise recommendations. 
              Upon completion of this study, we will conduct a lottery in which the winner will receive 
              a video-on-demand (VOD) rental of their highest-rated recommended movie as a reward.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-12">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-center w-16 h-16 bg-amber-500 bg-opacity-20 rounded-full mb-4 mx-auto">
              <Star className="text-amber-400" size={32} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Step 1.<br /> Rate Movies</h3>
            <p className="text-gray-400">
              Start by rating movies to help us understand your taste preferences. You can watch the trailer by clicking on the poster.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-center w-16 h-16 bg-amber-500 bg-opacity-20 rounded-full mb-4 mx-auto">
              <Target className="text-amber-400" size={32} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Step 2.<br /> Get Recommendations</h3>
            <p className="text-gray-400">
              Receive personalized movie recommendations based on your ratings and preferences.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-center w-16 h-16 bg-amber-500 bg-opacity-20 rounded-full mb-4 mx-auto">
              <Award className="text-amber-400" size={32} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Step 3.<br /> Evaluate Quality</h3>
            <p className="text-gray-400">
              Rate the recommended movies, as well as the diversity, novelty, and serendipity of our recommendations.
            </p>
          </div>
        </div>

        <div className="bg-amber-500 bg-opacity-10 border border-amber-500 rounded-xl p-8 mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">How It Works</h2>
          <div className="grid md:grid-cols-2 gap-6 text-left">
            <div>
              <h4 className="text-amber-400 font-semibold mb-2">Phase 1: Initial Ratings</h4>
              <p className="text-gray-300 text-sm">
                Rate movies from our curated collection. You'll need to rate a minimum number of movies 
                to proceed, but rating more movies improves recommendation accuracy.
              </p>
            </div>
            <div>
              <h4 className="text-amber-400 font-semibold mb-2">Phase 2: Personalized Recommendations</h4>
              <p className="text-gray-300 text-sm">
                Based on your ratings, we'll show you personalized recommendations. Rate all of these movies and 
                answer additional questions about diversity, novelty, and serendipity.
              </p>
            </div>
          </div>
        </div>

        <div className="text-center">
          <button
            onClick={onStart}
            className="bg-amber-500 hover:bg-amber-600 text-black font-bold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl inline-flex items-center gap-3 text-lg"
          >
            Start Rating Movies
            <ArrowRight size={24} />
          </button>
          <p className="text-gray-400 text-sm mt-4">
            This study takes approximately 40-60 minutes to complete
          </p>
        </div>
      </div>
    </div>
  );
};
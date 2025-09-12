import React, { useState } from 'react';
import { QuestionnaireData } from '../types';
import { CheckCircle, User, Globe, Briefcase, MessageSquare, Mail } from 'lucide-react';

interface QuestionnaireScreenProps {
  onComplete: (data: QuestionnaireData) => void;
}

export const QuestionnaireScreen: React.FC<QuestionnaireScreenProps> = ({ onComplete }) => {
  const [formData, setFormData] = useState<QuestionnaireData>({
    movieWatchingFrequency: '',
    streamingServices: [],
    primaryStreamingService: '',
    movieGenrePreferences: [],
    opennessToExperience: '',
    riskAversion: '',
    movieExpertise: '',
    attentionCheck: '',
    serendipityAttitude: '',
    noveltyAttitude: '',
    diversityAttitude: '',
    gender: '',
    ageRange: '',
    nationality: '',
    occupation: '',
    additionalComments: '',
    email: ''
  });

  const [errors, setErrors] = useState<string[]>([]);

  const movieFrequencyOptions = [
    '0-1',
    '2-3',
    '4-6',
    '7-10',
    'More than 10'
  ];


  const genreOptions = [
    'Action',
    'Adventure',
    'Animation',
    'Comedy',
    'Drama',
    'Horror',
    'Romance',
    'Science Fiction'
  ];

  const ageRangeOptions = [
    '18-24',
    '25-34',
    '35-44',
    '45-54',
    '55-64',
    '65 or older'
  ];

  const opennessOptions = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7'
  ];

  const riskAversionOptions = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7'
  ];

  const expertiseOptions = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7'
  ];

  const attitudeOptions = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7'
  ];

  const handleStreamingServiceChange = (service: string, checked: boolean) => {
    if (checked) {
      setFormData(prev => ({
        ...prev,
        streamingServices: [...prev.streamingServices, service]
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        streamingServices: prev.streamingServices.filter(s => s !== service)
      }));
    }
  };

  const handleGenreChange = (genre: string, checked: boolean) => {
    // For single selection, replace the array with the selected genre
    setFormData(prev => ({
      ...prev,
      movieGenrePreferences: checked ? [genre] : []
    }));
  };

  const validateForm = () => {
    const newErrors: string[] = [];

    if (!formData.movieWatchingFrequency) {
      newErrors.push('Movie watching frequency is required');
    }


    if (formData.movieGenrePreferences.length === 0) {
      newErrors.push('At least one movie genre preference is required');
    }

    if (!formData.opennessToExperience) {
      newErrors.push('Openness to experience question is required');
    }

    if (!formData.riskAversion) {
      newErrors.push('Risk aversion question is required');
    }

    if (!formData.movieExpertise) {
      newErrors.push('Movie expertise question is required');
    }

    if (!formData.serendipityAttitude) {
      newErrors.push('Serendipity attitude question is required');
    }

    if (!formData.noveltyAttitude) {
      newErrors.push('Novelty attitude question is required');
    }

    if (!formData.diversityAttitude) {
      newErrors.push('Diversity attitude question is required');
    }

    if (!formData.attentionCheck) {
      newErrors.push('Please answer the attention check question');
    }

    setErrors(newErrors);
    return newErrors.length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onComplete(formData);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-amber-500 rounded-full mb-4">
            <CheckCircle size={32} className="text-black" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Final Questionnaire</h1>
          <p className="text-gray-300">
            Please answer a few questions about your movie preferences and background
          </p>
        </div>

        {errors.length > 0 && (
          <div className="bg-red-900 border border-red-500 rounded-lg p-4 mb-6">
            <h3 className="text-red-400 font-semibold mb-2">Please complete the following required fields:</h3>
            <ul className="text-red-300 text-sm space-y-1">
              {errors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Mandatory Questions */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-6">Required Information</h2>
            
            {/* Movie Watching Frequency */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                How many movies do you watch per month? *
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {movieFrequencyOptions.map(option => (
                  <label key={option} className="flex items-center">
                    <input
                      type="radio"
                      name="movieFrequency"
                      value={option}
                      checked={formData.movieWatchingFrequency === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, movieWatchingFrequency: e.target.value }))}
                      className="mr-3 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300">{option}</span>
                  </label>
                ))}
              </div>
            </div>


            {/* Movie Genre Preferences */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                What is your most favorite movie genre? (Select one) *
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {genreOptions.map(genre => (
                  <label key={genre} className="flex items-center">
                    <input
                      type="radio"
                      name="movieGenre"
                      value={genre}
                      checked={formData.movieGenrePreferences.includes(genre)}
                      onChange={(e) => handleGenreChange(genre, e.target.checked)}
                      className="mr-3 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300">{genre}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Movie Expertise */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                How knowledgeable are you about movies in general? (1-Not at all knowledgeable, 7-Extremely knowledgeable) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {expertiseOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="movieExpertise"
                      value={option}
                      checked={formData.movieExpertise === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, movieExpertise: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Attention Check */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                To help us with the survey, please select "4" for this question. *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {['1', '2', '3', '4', '5', '6', '7'].map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="attentionCheck"
                      value={option}
                      checked={formData.attentionCheck === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, attentionCheck: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Openness to Experience */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                I enjoy trying new and different things. (1-Strongly disagree, 7-Strongly agree) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {opennessOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="opennessToExperience"
                      value={option}
                      checked={formData.opennessToExperience === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, opennessToExperience: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Risk Aversion */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                I prefer predictable and familiar experiences over unexpected ones. (1-Strongly disagree, 7-Strongly agree) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {riskAversionOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="riskAversion"
                      value={option}
                      checked={formData.riskAversion === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, riskAversion: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>


            {/* Serendipity Attitude */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                I enjoy receiving recommendations that are unexpected and surprising. (1-Strongly disagree, 7-Strongly agree) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {attitudeOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="serendipityAttitude"
                      value={option}
                      checked={formData.serendipityAttitude === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, serendipityAttitude: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Novelty Attitude */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                I prefer recommendations that are new and unfamiliar to me. (1-Strongly disagree, 7-Strongly agree) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {attitudeOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="noveltyAttitude"
                      value={option}
                      checked={formData.noveltyAttitude === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, noveltyAttitude: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Diversity Attitude */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">
                I like recommendations that include a variety of different genres and styles. (1-Strongly disagree, 7-Strongly agree) *
              </label>
              <div className="grid grid-cols-7 gap-2">
                {attitudeOptions.map(option => (
                  <label key={option} className="flex flex-col items-center">
                    <input
                      type="radio"
                      name="diversityAttitude"
                      value={option}
                      checked={formData.diversityAttitude === option}
                      onChange={(e) => setFormData(prev => ({ ...prev, diversityAttitude: e.target.value }))}
                      className="mb-1 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-gray-300 text-sm">{option}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Optional Questions */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
              <User className="mr-2" size={24} />
              Optional Information
            </h2>
            
            {/* Gender */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">Gender (Optional)</label>
              <select
                value={formData.gender || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, gender: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-600 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Non-binary">Non-binary</option>
                <option value="Prefer not to say">Prefer not to say</option>
                <option value="Other">Other</option>
              </select>
            </div>

            {/* Age Range */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3">Age Range (Optional)</label>
              <select
                value={formData.ageRange || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, ageRange: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-600 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                <option value="">Select age range</option>
                {ageRangeOptions.map(range => (
                  <option key={range} value={range}>{range}</option>
                ))}
              </select>
            </div>


            {/* Email for Lottery */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3 flex items-center">
                <Mail className="mr-2" size={20} />
                Email Address for Lottery Participation (Optional)
              </label>
              <input
                type="email"
                value={formData.email || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                placeholder="Enter your email address to participate in the lottery"
                className="w-full bg-gray-800 border border-gray-600 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
              <p className="text-gray-400 text-sm mt-2">
                If you provide your email, you'll be eligible for the lottery to win a video-on-demand (VOD) rental of your highest-rated recommended movie.
              </p>
            </div>

            {/* Additional Comments */}
            <div className="mb-6">
              <label className="block text-amber-400 font-medium mb-3 flex items-center">
                <MessageSquare className="mr-2" size={20} />
                Additional Comments (Optional)
              </label>
              <textarea
                value={formData.additionalComments || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, additionalComments: e.target.value }))}
                placeholder="Any additional thoughts about the study or your movie preferences..."
                rows={4}
                className="w-full bg-gray-800 border border-gray-600 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-amber-500 resize-vertical"
              />
            </div>
          </div>

          <div className="text-center">
            <button
              type="submit"
              className="bg-amber-500 hover:bg-amber-600 text-black font-bold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg text-lg"
            >
              Complete Study
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
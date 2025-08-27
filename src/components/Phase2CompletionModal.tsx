import React from 'react';
import { CheckCircle, Edit, ArrowRight } from 'lucide-react';

interface Phase2CompletionModalProps {
  isOpen: boolean;
  onModifyRatings: () => void;
  onFinish: () => void;
}

export const Phase2CompletionModal: React.FC<Phase2CompletionModalProps> = ({
  isOpen,
  onModifyRatings,
  onFinish
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl max-w-md w-full p-8 border border-gray-800">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500 rounded-full mb-6">
            <CheckCircle size={32} className="text-white" />
          </div>
          
          <h2 className="text-2xl font-bold text-white mb-4">
            Phase 2 Complete!
          </h2>
          
          <p className="text-gray-300 mb-8">
            You have successfully rated all recommended movies and answered the additional questions. 
            You can now finish the study or modify your ratings if needed.
          </p>
          
          <div className="space-y-4">
            <button
              onClick={onModifyRatings}
              className="w-full bg-gray-800 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 border border-gray-600 flex items-center justify-center gap-2"
            >
              <Edit size={20} />
              Modify Ratings
            </button>
            
            <button
              onClick={onFinish}
              className="w-full bg-amber-500 hover:bg-amber-600 text-black font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
            >
              Finish Study
              <ArrowRight size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
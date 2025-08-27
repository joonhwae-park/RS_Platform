import React from 'react';

interface SelectRatingProps {
  rating: number;
  onRatingChange: (rating: number) => void;
  label?: string;
  readonly?: boolean;
}

export const SelectRating: React.FC<SelectRatingProps> = ({ 
  rating, 
  onRatingChange, 
  label = "Rating",
  readonly = false 
}) => {
  const options = Array.from({ length: 11 }, (_, index) => index);

  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-amber-400 font-medium text-sm">
          {label}
        </label>
      )}
      <div className="flex items-center gap-3">
        <select
          value={rating === -1 ? '' : rating} // Show empty for unrated (-1)
          onChange={(e) => !readonly && onRatingChange(e.target.value === '' ? -1 : Number(e.target.value))}
          disabled={readonly}
          className="bg-gray-800 border border-gray-600 text-white rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all duration-200 min-w-[80px]"
        >
          <option value="">Select</option>
          {options.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <span className="text-gray-400 text-sm">
          {rating >= 0 ? `${rating}/10` : 'Not rated'}
        </span>
      </div>
    </div>
  );
};
import React, { useState } from 'react'
import PredictionResult from './PredictionResult'
import { 
  Calendar, 
  Bed, 
  Bath, 
  Home, 
  Map, 
  Square, 
  Star, 
  Bookmark, 
  Thermometer, 
  Hammer, 
  Car, 
  Building2, 
  PenTool, 
  Database,
  FileSpreadsheet,
  WavesLadder
} from 'lucide-react'

const NEIGHBORHOODS = [
  'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
  'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 
  'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 
  'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
]

const HOUSE_STYLES = [
  '1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'
]

const QUALITY_RATINGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
const CONDITION_RATINGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Sample data for quick testing
const SAMPLE_DATA = {
  LotArea: 8450,
  YearBuilt: 2003,
  FirstFlrSF: 856,
  SecondFlrSF: 854,
  TotalBsmtSF: 856,
  FullBath: 2,
  HalfBath: 1,
  BedroomAbvGr: 3,
  TotRmsAbvGrd: 8,
  Fireplaces: 0,
  GarageArea: 548,
  PoolArea: 0,
  Neighborhood: 'CollgCr',
  HouseStyle: '2Story',
  OverallQual: 7,
  OverallCond: 5
}

function HousePriceForm() {
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [formData, setFormData] = useState({...SAMPLE_DATA})

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: name === 'Neighborhood' || name === 'HouseStyle' 
        ? value 
        : parseInt(value, 10)
    })
  }

  const fillDummyData = () => {
    setFormData({...SAMPLE_DATA})
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      // Fix field names to match backend expectations
      const requestData = {
        data: [{
          ...formData,
          "1stFlrSF": formData.FirstFlrSF,
          "2ndFlrSF": formData.SecondFlrSF
        }]
      }
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })
      
      const result = await response.json()
      setPrediction(result)
    } catch (error) {
      console.error('Error predicting house price:', error)
      alert('Failed to predict house price. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const FormField = ({ label, name, value, onChange, type = "number", icon, min, max, options }) => {
    const InputIcon = () => (
      <div className="flex items-center ">
        {icon}
      </div>
    )

    if (options) {
      return (
        <div className="form-control w-full">
          <label className="label">
            <InputIcon />
            <span className="label-text text-gray-700 font-medium">{label}</span>
          </label>
          <div className="relative">
            
            <select
              name={name}
              value={value}
              onChange={onChange}
              className="select select-bordered w-full pl-10 bg-white border-gray-300 text-gray-700 focus:border-blue-500 focus:ring-blue-500"
            >
              {options.map(option => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        </div>
      )
    }
    
    return (
      <div className="form-control w-full">
        <label className="label">
          <InputIcon />
          <span className="label-text text-gray-700 font-medium">{label}</span>
        </label>
        <div className="relative">
          <input
            type={type}
            name={name}
            value={value}
            onChange={onChange}
            min={min}
            max={max}
            className="input input-bordered w-full pl-10 bg-white border-gray-300 text-gray-700 focus:border-blue-500 focus:ring-blue-500"
            required
          />
        </div>
      </div>
    )
  }

  return (
    <div className="card bg-white shadow-xl">
      <div className="card-body p-6 md:p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="card-title text-2xl text-gray-800 font-bold flex items-center gap-2">
            <FileSpreadsheet className="h-6 w-6 text-blue-500" />
            House Price Prediction
          </h2>
          
          <button 
            onClick={fillDummyData} 
            className="btn btn-sm bg-gray-100 hover:bg-gray-200 border-gray-300 text-gray-700 gap-2"
          >
            <Database className="h-4 w-4" />
            Fill Sample Data
          </button>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4">
            <FormField 
              label="Lot Area (sqft)" 
              name="LotArea" 
              value={formData.LotArea} 
              onChange={handleChange}
              icon={<Square className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Year Built" 
              name="YearBuilt" 
              value={formData.YearBuilt} 
              onChange={handleChange}
              min="1800"
              max="2025"
              icon={<Calendar className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="1st Floor Area (sqft)" 
              name="FirstFlrSF" 
              value={formData.FirstFlrSF} 
              onChange={handleChange}
              icon={<Home className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="2nd Floor Area (sqft)" 
              name="SecondFlrSF" 
              value={formData.SecondFlrSF} 
              onChange={handleChange}
              icon={<Building2 className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Basement Area (sqft)" 
              name="TotalBsmtSF" 
              value={formData.TotalBsmtSF} 
              onChange={handleChange}
              icon={<Bookmark className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Full Bathrooms" 
              name="FullBath" 
              value={formData.FullBath} 
              onChange={handleChange}
              min="0"
              icon={<Bath className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Half Bathrooms" 
              name="HalfBath" 
              value={formData.HalfBath} 
              onChange={handleChange}
              min="0"
              icon={<Thermometer className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Bedrooms" 
              name="BedroomAbvGr" 
              value={formData.BedroomAbvGr} 
              onChange={handleChange}
              min="0"
              icon={<Bed className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Total Rooms" 
              name="TotRmsAbvGrd" 
              value={formData.TotRmsAbvGrd} 
              onChange={handleChange}
              min="0"
              icon={<Home className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Fireplaces" 
              name="Fireplaces" 
              value={formData.Fireplaces} 
              onChange={handleChange}
              min="0"
              icon={<Thermometer className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Garage Area (sqft)" 
              name="GarageArea" 
              value={formData.GarageArea} 
              onChange={handleChange}
              min="0"
              icon={<Car className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Pool Area (sqft)" 
              name="PoolArea" 
              value={formData.PoolArea} 
              onChange={handleChange}
              min="0"
              icon={<WavesLadder className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Neighborhood" 
              name="Neighborhood" 
              value={formData.Neighborhood} 
              onChange={handleChange}
              options={NEIGHBORHOODS}
              icon={<Map className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="House Style" 
              name="HouseStyle" 
              value={formData.HouseStyle} 
              onChange={handleChange}
              options={HOUSE_STYLES}
              icon={<PenTool className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Overall Quality" 
              name="OverallQual" 
              value={formData.OverallQual} 
              onChange={handleChange}
              options={QUALITY_RATINGS}
              icon={<Star className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Overall Condition" 
              name="OverallCond" 
              value={formData.OverallCond} 
              onChange={handleChange}
              options={CONDITION_RATINGS}
              icon={<Hammer className="h-5 w-5 text-blue-400" />}
            />
          </div>

          <div className="mt-8">
            <button 
              type="submit" 
              className="btn w-full bg-blue-500 hover:bg-blue-600 text-white border-none"
              disabled={loading}
            >
              {loading ? 
                <span className="flex items-center gap-2">
                  <span className="loading loading-spinner"></span>
                  Computing Prediction...
                </span> 
                : 
                <span className="flex items-center gap-2">
                  <Star className="h-5 w-5" />
                  Predict House Price
                </span>
              }
            </button>
          </div>
        </form>
        
        {prediction && <PredictionResult prediction={prediction} />}
      </div>
    </div>
  )
}

export default HousePriceForm
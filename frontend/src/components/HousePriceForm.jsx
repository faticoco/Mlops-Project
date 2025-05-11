import React, { useState } from 'react'
import PredictionResult from './PredictionResult'
import { predictWithCsvRow } from '../api/api'
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

const SAMPLE_DATA = {
  LotArea: 12000,
  OverallQual: 9,
  OverallCond: 8,
  YearBuilt: 2020,
  YearRemodAdd: 2020,
  GrLivArea: 3000,
  FullBath: 3,
  HalfBath: 1,
  BedroomAbvGr: 4,
  KitchenAbvGr: 1,
  GarageCars: 3,
  GarageArea: 650,
  TotalBsmtSF: 1500,
  FirstFlrSF: 1600,
  SecondFlrSF: 1400,
  OpenPorchSF: 80,
  WoodDeckSF: 200
}

function HousePriceForm() {
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [formData, setFormData] = useState({})

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const requestData = {
        houses: [formData]
      }
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })
      
      const result = await response.json()
      setPrediction({
        prediction: result.predictions[0],
        formattedPrice: result.formatted_predictions[0]
      })
    } catch (error) {
      console.error('Error predicting house price:', error)
      alert('Failed to predict house price. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: parseInt(value, 10)
    })
  }

  const fillDummyData = () => {
    setFormData({...SAMPLE_DATA})
  }

  const FormField = ({ label, name, value, onChange, type = "number", icon, min, max }) => {
    const InputIcon = () => (
      <div className="flex items-center ">
        {icon}
      </div>
    )
    
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
              label="Overall Quality (1-10)" 
              name="OverallQual" 
              value={formData.OverallQual}
              onChange={handleChange}
              min="1"
              max="10" 
              icon={<Star className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Overall Condition (1-10)" 
              name="OverallCond" 
              value={formData.OverallCond}
              onChange={handleChange}
              min="1"
              max="10"
              icon={<Hammer className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Year Built" 
              name="YearBuilt" 
              value={formData.YearBuilt}
              onChange={handleChange}
              min="1800"
              max="2023"
              icon={<Calendar className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Year Remodeled" 
              name="YearRemodAdd" 
              value={formData.YearRemodAdd}
              onChange={handleChange}
              min="1800"
              max="2023"
              icon={<Hammer className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Above Ground Living Area (sqft)" 
              name="GrLivArea" 
              value={formData.GrLivArea}
              onChange={handleChange}
              icon={<Home className="h-5 w-5 text-blue-400" />}
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
              icon={<Bath className="h-5 w-5 text-blue-400" />}
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
              label="Kitchens" 
              name="KitchenAbvGr" 
              value={formData.KitchenAbvGr}
              onChange={handleChange}
              min="0"
              icon={<Home className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Garage Cars" 
              name="GarageCars" 
              value={formData.GarageCars}
              onChange={handleChange}
              min="0"
              icon={<Car className="h-5 w-5 text-blue-400" />}
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
              label="Total Basement Area (sqft)" 
              name="TotalBsmtSF" 
              value={formData.TotalBsmtSF}
              onChange={handleChange}
              icon={<Bookmark className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="First Floor Area (sqft)" 
              name="FirstFlrSF" 
              value={formData.FirstFlrSF}
              onChange={handleChange}
              icon={<Building2 className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Second Floor Area (sqft)" 
              name="SecondFlrSF" 
              value={formData.SecondFlrSF}
              onChange={handleChange}
              icon={<Building2 className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Open Porch Area (sqft)" 
              name="OpenPorchSF" 
              value={formData.OpenPorchSF}
              onChange={handleChange}
              min="0"
              icon={<Home className="h-5 w-5 text-blue-400" />}
            />

            <FormField 
              label="Wood Deck Area (sqft)" 
              name="WoodDeckSF" 
              value={formData.WoodDeckSF}
              onChange={handleChange}
              min="0"
              icon={<Square className="h-5 w-5 text-blue-400" />}
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
                'Get Price Prediction'
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
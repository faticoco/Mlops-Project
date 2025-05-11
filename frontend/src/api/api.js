import axios from 'axios';

// Base URL for the API
const API_BASE_URL = 'http://localhost:8000';

// Function to check the API health
const checkApiHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    console.log('API Health Status:', response.data);
    return response.data;
  } catch (error) {
    console.error('API Health Check Error:', error);
    throw error;
  }
};

// Function to make a prediction with custom features
const predictHousingPrice = async (features, featureNames = null) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      features: features,
      feature_names: featureNames
    });
    console.log('Prediction Result:', response.data);
    return response.data;
  } catch (error) {
    console.error('Prediction Error:', error);
    throw error;
  }
};

// Function to make a prediction with a CSV row
const predictWithCsvRow = async (csvRow) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict-csv-row`, csvRow, {
      headers: {
        'Content-Type': 'text/plain'
      }
    });
    console.log('CSV Row Prediction Result:', response.data);
    return response.data;
  } catch (error) {
    console.error('CSV Prediction Error:', error);
    throw error;
  }
};


const testApi = async () => {
  // Test API health
  await checkApiHealth();
  
  // Example with custom features (based on the example from your model file)
  const sampleFeatures = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23];
  const featureNames = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'];
  
  await predictHousingPrice(sampleFeatures, featureNames);
  
  // Example with CSV row from your message
  const csvRowSample = '1,60,RL,65.0,8450,Pave,Reg,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2003,2003,Gable,CompShg,VinylSd,VinylSd,BrkFace,196.0,Gd,TA,PConc,Gd,TA,No,GLQ,706,Unf,0,150,856,GasA,Ex,Y,SBrkr,856,854,0,1710,1,0,2,1,3,1,Gd,8,Typ,0,,Attchd,2003.0,RFn,2,548,TA,TA,Y,0,61,0,0,0,0,0,2,2008,WD,Normal,208500';
  
  await predictWithCsvRow(csvRowSample);
};


export {testApi,predictWithCsvRow,predictHousingPrice, checkApiHealth}
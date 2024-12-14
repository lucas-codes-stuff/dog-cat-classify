import './App.css';
import { FileUploader } from 'react-drag-drop-files';
import { useState } from 'react';

const fileTypes = ['JPG', 'PNG', 'JPEG'];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  interface PredictionResponse {
    class: string;
    confidence: string;
  }

  const handleChange = (file: File) => {
    setFile(file);
  };

  const predict = async () => {
    //check if file is null
    if (!file) {
      alert('Please upload a file');
      return;
    }

    //call the predict endpoint with the file
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('https://Lucas-F-cat-dog-classify.hf.space/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setPrediction(data);
  };

  return (
    <div className="bg-gradient-to-tr from-cyan-50 to-blue-50 w-full h-full flex flex-col justify-center items-center">
      <h1 className="text-3xl font-semibold mb-3">Dog or Cat Classifier</h1>
      <h3 className="mb-5">This will give a percentage amount of prediction</h3>
      <FileUploader
        handleChange={handleChange}
        name="file"
        types={fileTypes}
        label="Drop an image of a cat or dog here"
      />
      <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4" onClick={predict}>
        Predict Dog or Cat
      </button>
      {prediction && (
        <div className="mt-5 flex flex-col justify-center items-center">
          <h2 className="text-2xl font-semibold">Prediction</h2>
          <p className="text-lg">Class: {prediction.class}</p>
          <p className="text-lg">Percentage: {prediction.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;

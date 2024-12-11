import './App.css';
import { FileUploader } from 'react-drag-drop-files';
import { useState } from 'react';

const fileTypes = ['JPG'];

function App() {
  const [file, setFile] = useState<File | null>(null);
  interface File {
    name: string;
    size: number;
    type: string;
    d;
  }

  const handleChange = (file: File) => {
    setFile(file);
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
      <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4">
        Predict Dog or Cat
      </button>
    </div>
  );
}

export default App;

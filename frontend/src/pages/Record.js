import React, { useRef, useState } from "react";
import "../styles/soundRecorder.css";

const SoundRecorderLayout = () => {
  const fileInputRef = useRef(null);
  const [fileName, setFileName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [sounds, setSounds] = useState([]);
  const [stream, setStream] = useState(null);
  const mediaRecorderRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [jsonResponse, setJsonResponse] = useState(null);

  // Start or stop recording
  const handleRecord = () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsActive(false);

      mediaRecorderRef.current.ondataavailable = (event) => {
        const recordedSound = {
          name: `Sound ${sounds.length + 1}`,
          date: new Date().toISOString().split("T")[0],
          url: URL.createObjectURL(event.data),
        };
        setAudioBlob(event.data);
        setSounds([...sounds, recordedSound]);
      };

      // Stop all audio tracks when recording is done
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    } else {
      // Start recording
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((newStream) => {
          setStream(newStream);
          mediaRecorderRef.current = new MediaRecorder(newStream);
          mediaRecorderRef.current.start();
          setIsRecording(true);
          setIsActive(true);
        })
        .catch((error) => console.error("Error accessing microphone:", error));
    }
  };

  // Handle file selection and upload
  const handleFileInputClick = () => fileInputRef.current.click();

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
    } else {
      setFileName("");
    }
  };

  // Handle file upload to backend
  const handleSubmit = async () => {
    const file = fileInputRef.current.files[0];
    if (!file) {
      alert("Please select a .wav file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    console.log("Uploading file:", file.name);

    try {
      const response = await fetch("http://localhost:5000/detect-noise", {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        throw new Error("Error detecting noise");
      }

      const result = await response.json();
      console.log("Response from backend:", result);
      setJsonResponse(result);

      //   const uploadedSound = {
      //     name: file.name,
      //     date: new Date().toISOString().split("T")[0],
      //     url: `http://localhost:5000/uploads/${file.name}`,
      //   };
      //   setSounds((prevSounds) => [...prevSounds, uploadedSound]);
      // } catch (error) {
      //   console.error("Error:", error);
      // }

      const processedSound = {
        name: `Processed_${file.name}`,
        date: new Date().toISOString().split("T")[0],
        url: result.file_url, // Modified file URL
      };
      setSounds((prevSounds) => [...prevSounds, processedSound]);
    } catch (error) {
      console.error("Error:", error);
      alert("There was an error uploading the file. Please try again.");
    }
  };

  return (
    <div className="layout-container">
      {/* Recorded Sounds List */}
      <div className="sounds-list">
        <h2>All Recorded Sounds</h2>
        {sounds.length > 0 ? (
          sounds.map((sound, index) => (
            <div key={index} className="sound-card">
              <div className="card-content">
                <span className="sound-name">{sound.name}</span>
              </div>
              <audio controls src={sound.url} className="audio-player" />
            </div>
          ))
        ) : (
          <p>No sounds recorded yet.</p>
        )}
      </div>

      {/* Microphone and Upload Section */}
      <div className="mic-section">
        <div className="options-container">
          {/* Microphone recording section */}
          <div className="mic">
            <div className="spinner-container">
              <div
                className={`spinner ${isActive ? "active" : ""}`}
                onClick={handleRecord}
              >
                <div className="spinner1"></div>
              </div>
              <div className="status-message">
                {isRecording
                  ? "Recording..."
                  : audioBlob
                  ? "Recorded"
                  : "Click to Record"}
              </div>
            </div>
          </div>

          <div className="divider"></div>

          {/* File upload section */}
          <div className="file-upload-section">
            <h2>Upload a File</h2>
            <div
              className="plusButton"
              onClick={handleFileInputClick}
              role="button"
              aria-label="Upload File"
            >
              <svg
                className="plusIcon"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 30 30"
              >
                <path d="M13.75 23.75V16.25H6.25V13.75H13.75V6.25H16.25V13.75H23.75V16.25H16.25V23.75H13.75Z"></path>
              </svg>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            {fileName && <p>Selected File: {fileName}</p>}

            <button
              style={{
                width: "100%",
                marginTop: "20px",
              }}
              onClick={handleSubmit}
            >
              Submit
            </button>

            {/* JSON response display */}
            {jsonResponse && (
              // <div className="json-response" style={{ display: "none" }}>
              //   <h3>Detected Noise Segments:</h3>
              //   <pre>{JSON.stringify(jsonResponse, null, 2)}</pre>
              // </div>
              <div className="json-response" style={{ display: "none" }}>
                <h3>Detected Noise Segments:</h3>
                <pre>{JSON.stringify(jsonResponse, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SoundRecorderLayout;

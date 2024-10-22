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

  const handleClick = () => {
    if (isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsActive(false);
      mediaRecorderRef.current.ondataavailable = (event) => {
        setAudioBlob(event.data);
        const newSound = {
          name: `Sound ${sounds.length + 1}`,
          date: new Date().toISOString().split("T")[0],
          url: URL.createObjectURL(event.data),
        };
        setSounds([...sounds, newSound]);
      };
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    } else {
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

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

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

    try {
      const response = await fetch("http://localhost:5000/detect-noise", {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Error in detecting noise");
      }

      const result = await response.json();
      setJsonResponse(result);

      const newSound = {
        name: file.name,
        date: new Date().toISOString().split("T")[0],
        url: `http://localhost:5000/uploads/${file.name}`,
      };
      setSounds((prevSounds) => [...prevSounds, newSound]);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="layout-container">
      <div className="sounds-list">
        <h2>All Recorded Sounds</h2>
        {sounds.map((sound, index) => (
          <div key={index} className="sound-card">
            <div className="card-content">
              <span className="sound-name">{sound.name}</span>
            </div>
            {/* Directly include the audio player with controls */}
            <audio controls src={sound.url} className="audio-player" />
          </div>
        ))}
      </div>

      <div className="mic-section">
        <div className="options-container">
          <div className="mic">
            <div className="spinner-container">
              <div
                className={`spinner ${isActive ? "active" : ""}`}
                onClick={handleClick}
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

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "30px",
              alignItems: "center",
              justifyContent: "center",
              height: "40vh",
              width: "14.6%",
            }}
          >
            <h2>Upload a File</h2>
            <div
              tabIndex="0"
              className="plusButton"
              onClick={handleButtonClick}
              role="button"
              aria-label="Upload File"
            >
              <svg
                className="plusIcon"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 30 30"
              >
                <g mask="url(#mask0_21_345)">
                  <path d="M13.75 23.75V16.25H6.25V13.75H13.75V6.25H16.25V13.75H23.75V16.25H16.25V23.75H13.75Z"></path>
                </g>
              </svg>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            {fileName && <p>Selected File: {fileName}</p>}

            <button style={{ width: "100%" }} onClick={handleSubmit}>
              <span className="span-mother">
                <span>S</span>&nbsp;
                <span>U</span>&nbsp;
                <span>B</span>&nbsp;
                <span>M</span>&nbsp;
                <span>I</span>&nbsp;
                <span>T</span>&nbsp;
              </span>
              <span className="span-mother2">
                <span>S</span>&nbsp;
                <span>U</span>&nbsp;
                <span>B</span>&nbsp;
                <span>M</span>&nbsp;
                <span>I</span>&nbsp;
                <span>T</span>&nbsp;
              </span>
            </button>

            {/* Display JSON response */}
            {jsonResponse && (
              <div className="json-response">
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

// import React, { useRef, useState, useEffect } from "react";
// import "../styles/soundRecorder.css";

// const SoundRecorderLayout = () => {
//   const fileInputRef = useRef(null);
//   const [fileName, setFileName] = useState("");
//   const [isRecording, setIsRecording] = useState(false);
//   const [audioBlob, setAudioBlob] = useState(null);
//   const [sounds, setSounds] = useState([]);
//   const [playingIndex, setPlayingIndex] = useState(null);
//   const mediaRecorderRef = useRef(null);
//   const audioRef = useRef(null);
//   const [isActive, setIsActive] = useState(false);
//   const [stream, setStream] = useState(null);
//   const [jsonResponse, setJsonResponse] = useState(null);

//   const handleClick = () => {
//     if (isRecording) {
//       mediaRecorderRef.current.stop();
//       setIsRecording(false);
//       setIsActive(false);
//       mediaRecorderRef.current.ondataavailable = (event) => {
//         setAudioBlob(event.data);
//         const newSound = {
//           name: `Sound ${sounds.length + 1}`,
//           date: new Date().toISOString().split("T")[0],
//           url: URL.createObjectURL(event.data),
//         };
//         setSounds([...sounds, newSound]);
//       };
//       if (stream) {
//         stream.getTracks().forEach((track) => track.stop());
//       }
//     } else {
//       navigator.mediaDevices
//         .getUserMedia({ audio: true })
//         .then((newStream) => {
//           setStream(newStream);
//           mediaRecorderRef.current = new MediaRecorder(newStream);
//           mediaRecorderRef.current.start();
//           setIsRecording(true);
//           setIsActive(true);
//         })
//         .catch((error) => console.error("Error accessing microphone:", error));
//     }
//   };

//   const handleButtonClick = () => {
//     fileInputRef.current.click();
//   };

//   const handlePlayPauseClick = (index) => {
//     if (playingIndex === index) {
//       if (audioRef.current.paused) {
//         audioRef.current.play();
//       } else {
//         audioRef.current.pause();
//       }
//     } else {
//       setPlayingIndex(index);
//     }
//   };

//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       setFileName(file.name);
//     } else {
//       setFileName("");
//     }
//   };

//   useEffect(() => {
//     if (audioRef.current) {
//       audioRef.current.onended = () => {
//         setPlayingIndex(null); // Reset playing index when audio ends
//       };
//       if (playingIndex !== null) {
//         audioRef.current.src = sounds[playingIndex].url;
//         audioRef.current
//           .play()
//           .catch((error) => console.error("Error playing audio:", error));
//       }
//     }
//   }, [playingIndex, sounds]);

//   // Handle file upload to backend
//   const handleSubmit = async () => {
//     const file = fileInputRef.current.files[0];
//     if (!file) {
//       alert("Please select a .wav file first!");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", file);

//     try {
//       const response = await fetch("http://localhost:5000/detect-noise", {
//         method: "POST",
//         body: formData,
//         headers: {
//           Accept: "application/json",
//         },
//       });

//       if (!response.ok) {
//         throw new Error("Error in detecting noise");
//       }

//       const result = await response.json();
//       setJsonResponse(result);

//       const newSound = {
//         name: file.name,
//         date: new Date().toISOString().split("T")[0],
//         url: `http://localhost:5000/uploads/${file.name}`,
//       };
//       setSounds((prevSounds) => [...prevSounds, newSound]);
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <div className="layout-container">
//       <div className="sounds-list">
//         <h2>All Recorded Sounds</h2>
//         {sounds.map((sound, index) => (
//           <div key={index} className="sound-card">
//             <div className="card-content">
//               <span className="sound-name">{sound.name}</span>
//             </div>
//             <audio controls src={sound.url} className="audio-player" />
//           </div>
//         ))}
//       </div>

//       <div className="mic-section">
//         <div className="options-container">
//           <div className="mic">
//             <div className="spinner-container">
//               <div
//                 className={`spinner ${isActive ? "active" : ""}`}
//                 onClick={handleClick}
//               >
//                 <div className="spinner1"></div>
//               </div>
//               <div className="status-message">
//                 {isRecording
//                   ? "Recording..."
//                   : audioBlob
//                   ? "Recorded"
//                   : "Click to Record"}
//               </div>
//             </div>
//           </div>

//           <div className="divider"></div>

//           <div
//             style={{
//               display: "flex",
//               flexDirection: "column",
//               gap: "30px",
//               alignItems: "center",
//               justifyContent: "center",
//               height: "40vh",
//               width: "14.6%",
//             }}
//           >
//             <h2>Upload a File</h2>
//             <div
//               tabIndex="0"
//               className="plusButton"
//               onClick={handleButtonClick}
//               role="button"
//               aria-label="Upload File"
//             >
//               <svg
//                 className="plusIcon"
//                 xmlns="http://www.w3.org/2000/svg"
//                 viewBox="0 0 30 30"
//               >
//                 <g mask="url(#mask0_21_345)">
//                   <path d="M13.75 23.75V16.25H6.25V13.75H13.75V6.25H16.25V13.75H23.75V16.25H16.25V23.75H13.75Z"></path>
//                 </g>
//               </svg>
//             </div>
//             <input
//               type="file"
//               ref={fileInputRef}
//               style={{ display: "none" }}
//               onChange={handleFileChange}
//             />
//             {fileName && <p>Selected File: {fileName}</p>}

//             <button style={{ width: "100%" }} onClick={handleSubmit}>
//               <span className="span-mother">
//                 <span>S</span>&nbsp;
//                 <span>U</span>&nbsp;
//                 <span>B</span>&nbsp;
//                 <span>M</span>&nbsp;
//                 <span>I</span>&nbsp;
//                 <span>T</span>&nbsp;
//               </span>
//               <span className="span-mother2">
//                 <span>S</span>&nbsp;
//                 <span>U</span>&nbsp;
//                 <span>B</span>&nbsp;
//                 <span>M</span>&nbsp;
//                 <span>I</span>&nbsp;
//                 <span>T</span>&nbsp;
//               </span>
//             </button>

//             {/* Display JSON response */}
//             {jsonResponse && (
//               <div className="json-response">
//                 <h3>Detected Noise Segments:</h3>
//                 <pre>{JSON.stringify(jsonResponse, null, 2)}</pre>
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default SoundRecorderLayout;

// import React, { useRef, useState, useEffect } from 'react';
// import '../styles/soundRecorder.css';

// const SoundRecorderLayout = () => {
//     const fileInputRef = useRef(null);
//     const [fileName, setFileName] = useState('');
//     const [isRecording, setIsRecording] = useState(false);
//     const [audioURL, setAudioURL] = useState('');
//     const [audioBlob, setAudioBlob] = useState(null);
//     const [sounds, setSounds] = useState([]);
//     const [playingIndex, setPlayingIndex] = useState(null);
//     const mediaRecorderRef = useRef(null);
//     const audioRef = useRef(null);
//     const [isActive, setIsActive] = useState(false);
//     const [stream, setStream] = useState(null);

//     const handleClick = () => {
//         if (isRecording) {
//             // Stop recording
//             mediaRecorderRef.current.stop();
//             setIsRecording(false);
//             setIsActive(false);
//             mediaRecorderRef.current.ondataavailable = (event) => {
//                 setAudioBlob(event.data);
//                 const newSound = {
//                     name: `Sound ${sounds.length + 1}`,
//                     date: new Date().toISOString().split('T')[0],
//                     url: URL.createObjectURL(event.data),
//                 };
//                 setSounds([...sounds, newSound]);
//                 setAudioURL(newSound.url);
//             };
//             // Stop the audio stream
//             if (stream) {
//                 stream.getTracks().forEach(track => track.stop());
//             }
//         } else {
//             // Start recording
//             navigator.mediaDevices.getUserMedia({ audio: true })
//                 .then(newStream => {
//                     setStream(newStream); // Save the stream
//                     mediaRecorderRef.current = new MediaRecorder(newStream);
//                     mediaRecorderRef.current.start();
//                     setIsRecording(true);
//                     setIsActive(true);
//                 })
//                 .catch(error => console.error('Error accessing microphone:', error));
//         }
//     };

//     const handlePlayPauseClick = (index) => {
//         if (playingIndex === index) {
//             if (audioRef.current.paused) {
//                 audioRef.current.play();
//             } else {
//                 audioRef.current.pause();
//             }
//         } else {
//             setPlayingIndex(index);
//         }
//     };

//     useEffect(() => {
//         if (audioRef.current) {
//             audioRef.current.onended = () => {
//                 setPlayingIndex(null); // Reset playing index when audio ends
//             };
//             if (playingIndex !== null) {
//                 audioRef.current.src = sounds[playingIndex].url;
//                 audioRef.current.play().catch(error => console.error('Error playing audio:', error));
//             }
//         }
//     }, [playingIndex]);

//     const handleButtonClick = () => {
//         fileInputRef.current.click();
//     };

//     const handleFileChange = (event) => {
//         const file = event.target.files[0];
//         if (file) {
//             const fileURL = URL.createObjectURL(file);
//             const newSound = {
//                 name: file.name,
//                 date: new Date().toISOString().split('T')[0],
//                 url: fileURL,
//             };
//             setSounds([...sounds, newSound]);
//             setFileName(file.name);
//         } else {
//             setFileName('');
//         }
//     };

//     return (
//         <div className="layout-container">
//             <div className="sounds-list">
//                 <h2>All Recorded Sounds</h2>
//                 {sounds.map((sound, index) => (
//                     <div key={index} className="sound-card">
//                         <div className="card-content">
//                             <span className="sound-name">{sound.name}</span>
//                             {/* <span className="sound-date">{sound.date}</span> */}
//                         </div>
//                         {/* Directly include the audio player here */}
//                         <audio
//                             controls
//                             src={sound.url}
//                             className="audio-player"
//                         />
//                     </div>
//                 ))}
//             </div>

//             <div className="mic-section">
//                 <div className="options-container">
//                     <div className="mic">
//                         <div className="spinner-container">
//                             <div
//                                 className={`spinner ${isActive ? 'active' : ''}`}
//                                 onClick={handleClick}
//                             >
//                                 <div className="spinner1"></div>
//                             </div>
//                             <div className="status-message">
//                                 {isRecording ? 'Recording...' : audioBlob ? 'Recorded' : 'Click to Record'}
//                             </div>
//                         </div>
//                     </div>
//                     <div className="divider"></div>
//                     <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', alignItems: 'center', justifyContent: 'center', height: '40vh', width: '14.6%' }}>
//                         <h2>Upload a File</h2>
//                         <div
//                             tabIndex="0"
//                             className="plusButton"
//                             onClick={handleButtonClick}
//                             role="button"
//                             aria-label="Upload File"
//                         >
//                             <svg className="plusIcon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30">
//                                 <g mask="url(#mask0_21_345)">
//                                     <path d="M13.75 23.75V16.25H6.25V13.75H13.75V6.25H16.25V13.75H23.75V16.25H16.25V23.75H13.75Z"></path>
//                                 </g>
//                             </svg>
//                         </div>
//                         <input
//                             type="file"
//                             ref={fileInputRef}
//                             style={{ display: 'none' }}
//                             onChange={handleFileChange}
//                         />
//                         <div>
//                             {fileName && <p>Selected File: {fileName}</p>}
//                         </div>

//                         <button style={{ width: '100%' }}>
//                             <span className="span-mother">
//                                 <span>S</span>&nbsp;
//                                 <span>U</span>&nbsp;
//                                 <span>B</span>&nbsp;
//                                 <span>M</span>&nbsp;
//                                 <span>I</span>&nbsp;
//                                 <span>T</span>&nbsp;
//                             </span>
//                             <span className="span-mother2">
//                                 <span>S</span>&nbsp;
//                                 <span>U</span>&nbsp;
//                                 <span>B</span>&nbsp;
//                                 <span>M</span>&nbsp;
//                                 <span>I</span>&nbsp;
//                                 <span>T</span>&nbsp;
//                             </span>
//                         </button>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default SoundRecorderLayout;

// correct code till now

// import React, { useRef, useState } from 'react';
// import '../styles/soundRecorder.css';

// const SoundRecorderLayout = () => {
//     const fileInputRef = useRef(null);
//     const [fileName, setFileName] = useState('');
//     const [isRecording, setIsRecording] = useState(false);
//     const [audioURL, setAudioURL] = useState('');
//     const [audioBlob, setAudioBlob] = useState(null);
//     const mediaRecorderRef = useRef(null);
//     const audioRef = useRef(null);
//     const [isActive, setIsActive] = useState(false);
//     const [stream, setStream] = useState(null);
//     const [soundList, setSoundList] = useState([]);

//     const handleClick = () => {
//         if (isRecording) {
//             // Stop recording
//             mediaRecorderRef.current.stop();
//             setIsRecording(false);
//             setIsActive(false);
//             mediaRecorderRef.current.ondataavailable = (event) => {
//                 setAudioBlob(event.data);
//                 const newAudioURL = URL.createObjectURL(event.data);
//                 setAudioURL(newAudioURL);
//                 // Add new recording to the sound list
//                 const date = new Date().toISOString().split('T')[0]; // Get current date
//                 setSoundList(prevList => [
//                     ...prevList,
//                     { name: `Sound ${prevList.length + 1}`, url: newAudioURL, date }
//                 ]);
//             };
//             // Stop the audio stream
//             if (stream) {
//                 stream.getTracks().forEach(track => track.stop());
//             }
//         } else {
//             // Start recording
//             navigator.mediaDevices.getUserMedia({ audio: true })
//                 .then(newStream => {
//                     setStream(newStream); // Save the stream
//                     mediaRecorderRef.current = new MediaRecorder(newStream);
//                     mediaRecorderRef.current.start();
//                     setIsRecording(true);
//                     setIsActive(true);
//                 })
//                 .catch(error => console.error('Error accessing microphone:', error));
//         }
//     };

//     const handlePlayPauseClick = (url) => {
//         const audio = new Audio(url);
//         if (audio.paused) {
//             audio.play();
//         } else {
//             audio.pause();
//         }
//     };

//     const handleButtonClick = () => {
//         fileInputRef.current.click();
//     };

//     const handleFileChange = (event) => {
//         const file = event.target.files[0];
//         if (file) {
//             const fileURL = URL.createObjectURL(file);
//             const date = new Date().toISOString().split('T')[0]; // Get current date
//             setSoundList(prevList => [
//                 ...prevList,
//                 { name: file.name, url: fileURL, date }
//             ]);
//             setFileName(file.name);
//         } else {
//             setFileName('');
//         }
//     };

//     return (
//         <div className="layout-container">
//             <div className="sounds-list">
//                 <h2>All Recorded Sounds</h2>
//                 {soundList.map((sound, index) => (
//                     <div key={index} className="sound-card">
//                         <div className="card-content">
//                             <span className="sound-name">{sound.name}</span>
//                             <span className="sound-date">{sound.date}</span>
//                         </div>
//                         <button
//                             className="play-pause-btn"
//                             onClick={() => handlePlayPauseClick(sound.url)}
//                         >
//                             Play
//                         </button>
//                     </div>
//                 ))}
//             </div>

//             <div className="mic-section">
//                 <div className="options-container">
//                     <div className="mic">
//                         <div className="spinner-container">
//                             <div
//                                 className={`spinner ${isActive ? 'active' : ''}`}
//                                 onClick={handleClick}
//                             >
//                                 <div className="spinner1"></div>
//                             </div>
//                             <div className="status-message">
//                                 {isRecording ? 'Recording...' : audioBlob ? 'Recorded' : 'Click to Record'}
//                             </div>
//                             {audioURL && (
//                                 <div className="recorded-voice">
//                                     <audio ref={audioRef} src={audioURL} controls />
//                                 </div>
//                             )}
//                         </div>
//                     </div>
//                     <div className="divider"></div>
//                     <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', alignItems: 'center', justifyContent: 'center', height: '40vh', width: '14.6%' }}>
//                         <div
//                             tabIndex="0"
//                             className="plusButton"
//                             onClick={handleButtonClick}
//                             role="button"
//                             aria-label="Upload File"
//                         >
//                             <svg className="plusIcon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30">
//                                 <g mask="url(#mask0_21_345)">
//                                     <path d="M13.75 23.75V16.25H6.25V13.75H13.75V6.25H16.25V13.75H23.75V16.25H16.25V23.75H13.75Z"></path>
//                                 </g>
//                             </svg>
//                         </div>
//                         <input
//                             type="file"
//                             ref={fileInputRef}
//                             style={{ display: 'none' }}
//                             onChange={handleFileChange}
//                         />
//                         <div>
//                             {fileName && <p>Selected File: {fileName}</p>}
//                         </div>

//                         <button style={{ width: '100%' }}>
//                             <span className="span-mother">
//                                 <span>S</span>&nbsp;
//                                 <span>U</span>&nbsp;
//                                 <span>B</span>&nbsp;
//                                 <span>M</span>&nbsp;
//                                 <span>I</span>&nbsp;
//                                 <span>T</span>&nbsp;
//                             </span>
//                             <span className="span-mother2">
//                                 <span>S</span>&nbsp;
//                                 <span>U</span>&nbsp;
//                                 <span>B</span>&nbsp;
//                                 <span>M</span>&nbsp;
//                                 <span>I</span>&nbsp;
//                                 <span>T</span>&nbsp;
//                             </span>
//                         </button>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default SoundRecorderLayout;

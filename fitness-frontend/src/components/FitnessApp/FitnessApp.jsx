import React from "react";
import { useState, useEffect, useRef } from "react";
import ReactPlayer from 'react-player';
import { useNavigate } from "react-router-dom";
import { Loader2 } from 'lucide-react';
import './FitnessApp.css';

const FitnessApp = () => {
    const videoRef = useRef(null);
    const [user, setUser] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [streamUrl, setStreamUrl] = useState(null);
    const [progressUrl, setProgressUrl] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isLoadingChatbot, setIsLoadingChatbot] = useState(false);
    const [exerciseName, setExerciseName] = useState("");
    const [exerciseWeight, setExerciseWeight] = useState(0);
    const [videoFile, setVideoFile] = useState(null);
    const [chatbotResponse, setChatbotResponse] = useState("");
    const navigateProgess = useNavigate();
    const navigateLogin = useNavigate();

    const exercise_type = [
        "bench press", "barbell biceps curl", "chest fly machine",
        "deadlift", "decline bench press", "hammer curl",
        "hip thrust", "incline bench press", "lat pulldown",
        "lateral raise", "leg extension", "leg raises",
        "plank", "pull up", "push-up", "romanian deadlift",
        "russian twist", "shoulder press", "squat", "t bar row",
        "tricep dips", "tricep pushdown"
    ]

    useEffect(() => {
        const fetchUser = async () => {
            const token = localStorage.getItem("token");
            if(!token) {
                alert("No token found. Redirecting to login...");
                navigateLogin("/");
                return;
            }

            try {
                const response = await fetch("http://127.0.0.1:8000/api/user", {
                    method: "GET",
                    headers: {
                        "Authorization": `Token ${token}`,
                        "Content-Type": "application/json"
                    },
                });
                if(!response.ok) {
                    throw new Error("Failed to fetch user data.");
                }

                const data = await response.json();
                setUser(data);
            } catch (error) {
                console.error("Error fetching user:", error);
            }
        };
        fetchUser();
    }, [navigateLogin]);



    const handleFileChange = (event) => {
        setVideoFile(event.target.files[0]);
    };

    const handleSignoffButtonClick = () => {
        navigateProgess("/");
    }


    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!videoFile) {
            console.error("No video file selected.");
            return;
        }
        if(!exerciseName) {
            console.error("No exercise selected.");
            return;
        }

        setIsAnalyzing(true);
        setIsLoadingChatbot(true);
        setChatbotResponse("");

        try {
            const formData = new FormData();
            formData.append('file', videoFile);
            formData.append('exercise_name', exerciseName);
            formData.append('exercise_weight', exerciseWeight);
            const token = localStorage.getItem('token');
            if(!token) {
                throw new Error("No token found. User might not be authenticated.");
            }
            const response = await fetch('http://127.0.0.1:8000/api/upload/', {
                method: 'POST',
                body: formData,
                headers: {
                    "Authorization": `Token ${localStorage.getItem('token')}`
                }
            })
            if(!response.ok) {
                throw new Error(`HTTP error! status ${response.status}`);
            }
            // After successful upload, potentially fetch the stream URL
            const streamData = await response.json();
            if(!streamData) {
                throw new Error("No stream URL returned from the backend.");
            }
            if(streamData.stream_url) {
                setStreamUrl(streamData.stream_url);
            }
            if(streamData.progress_url) {
                setProgressUrl(streamData.progress_url);
            }
            setIsAnalyzing(false);

            //Start polling for analysis status
            const exerciseId = streamData.exercise.id;
            const pollInterval = setInterval(async () => {
                const statusResponse = await fetch(
                    `http://127.0.0.1:8000/api/analysis-status/${exerciseId}/`,
                    {
                        headers: {
                            "Authorization": `Token ${localStorage.getItem('token')}`
                        }
                    }
                );
                const statusData = await statusResponse.json();

                if (statusData.status === 'completed') {
                    setChatbotResponse(statusData.chatbot_response);
                    setIsLoadingChatbot(false);
                    clearInterval(pollInterval);
                }
            }, 2000); //Poll every 2 seconds
        } catch(error) {
            console.error("Error uploading file:", error);
            setIsAnalyzing(false);
            setIsLoadingChatbot(false);
            alert("Error uploading file please try again (Make sure file is in .mp4 format).");
        }
    };

    return (
        <div className="fitness-app-container">
            <h1 className="fitness-app-title">Workout Video Upload</h1>
            <form className="exercise-form" onSubmit={handleSubmit}>
                <label htmlFor="exercises">Select your exercise:</label>
                <select className="exercise-dropdown" name="exercises_dropdown" value={exerciseName} required onChange={(e) => setExerciseName(e.target.value)}>
                    <option value="" disabled>Select an exercise</option>
                    {exercise_type.map((exercise_name, index) => (
                        <option key={index} value={exercise_name}>{exercise_name}</option>
                    ))}
                </select>
                <label>How much weight are you lifting? <input className="weight-input" onChange={(e) => setExerciseWeight(e.target.value)} required/></label>
                <label>Please select your mp4 workout video to analyze:<input className="file-upload" type="file" onChange={handleFileChange} required/></label>
                <button className="submit-button" type="submit" disabled={isAnalyzing || isLoadingChatbot}>{isAnalyzing ? 'Uploading...' : 'Upload'}</button>
            </form>

            {isAnalyzing && (
                <div className="analyzing-message">
                    <Loader2 className="loading-spinner" />
                    <p>Processing video... </p>
                </div>
            )}

            {streamUrl && (
                <div className="video-progress-container">
                    <ReactPlayer
                        url={streamUrl}
                        playing={isPlaying}
                        controls={true}
                        width="640px"
                        height="360px"
                        style={{ pointerEvents: 'auto'}}
                        className="video-container"
                    />
                    {progressUrl && (
                        <img className="progress-image" src={progressUrl} alt="Progress Snapshot"/>
                    )}
                </div>
            )}

            {isLoadingChatbot && (
                <div className="chatbot-loading">
                    <Loader2 className="loading-spinner" />
                    <p>Analyzing exercise form...</p>
                </div>
            )}

            {chatbotResponse && !isLoadingChatbot && (
                <div className="chatbot-response-container">
                    <h3>Exercise Analysis</h3>
                    <p className="chatbot-response">{chatbotResponse}</p>
                </div>    
            )}
            <button className="signoff-button" onClick={handleSignoffButtonClick}>Sign out</button>
        </div>
    );
};

export default FitnessApp;
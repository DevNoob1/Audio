import React from 'react';
import './App.css';
import './styles/navbar.css';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import Home from './pages/Home.js';
import Register from './pages/Register.js';
import Dashboard from './pages/Dashboard.js';
import Record from './pages/Record.js';
import Login from './pages/Login.js';
import Navbar from './pages/Navbar.js';
import About from './pages/About.js';

const App = () => {
  return (
    <div className='app'>
      <BrowserRouter>
        <ConditionalNavbar />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Home />} />
          <Route path="/record" element={<Record />} />
          <Route path="/about" element={<About />} />
          {/* Uncomment if you have Register page */}
          {/* <Route path="/register" element={<Register />} /> */}
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

const ConditionalNavbar = () => {
  const location = useLocation();

  return location.pathname !== '/login' && <Navbar />;
};

export default App;

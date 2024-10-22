import React, { useState } from 'react';
import '../styles/navbar.css';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <div className="navbar-container">
            <nav className="navbar">
                <div className="navitems">
                    <Link to="/" className="nav-item">Home</Link>
                    <Link to="/record" className="nav-item">Record</Link>
                    <Link to="/about" className="nav-item">About</Link>
                    <Link to="/login" className="nav-item">Logout</Link>
                    {/* <Link to="/services" className="nav-item">Services</Link>
                    <Link to="/dashboard" className="nav-item">Dashboard</Link> */}
                </div>
            </nav>
        </div>
    );
};

export default Navbar;

import React from 'react';
import '../styles/Home.css';
import { useState, useEffect } from 'react';
const Header = () => {
    const [text, setText] = useState('');
    const fullText = "NoisePro.AI addresses tinnitus, impacting 15% of the global population, with personalized sound therapy. It improves on current masking treatments and extends to general noise-suppressing solutions.";

    // useEffect(() => {
    //     let index = 0;
    //     const typeInterval = setInterval(() => {
    //         setText(fullText.slice(0, index + 1));
    //         index += 1;
    //         if (index === fullText.length) {
    //             clearInterval(typeInterval);
    //         }
    //     }, 50); // Adjust typing speed here (50ms per character)
    //     return () => clearInterval(typeInterval);
    // }, []);

    return (
        <div className="Home">
            <div className="homeleft">
                <h1>NoisePro.AI
                </h1>
                <div className="typewriter">
                    {fullText}
                </div>

            </div>
        </div>
    );
};

export default Header;

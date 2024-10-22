import React from 'react';
import '../styles/about.css'; // Import the CSS file

const About = () => {
    return (
        <div className="container3">
            <h1 className="title"><u>About This Project</u></h1>
            <p>
                This sound masking application is designed to offer a highly customizable and adaptive audio experience tailored to individual user needs. By leveraging the latest advancements in modern web technologies and integrating cutting-edge machine learning algorithms, the application empowers users to create and fine-tune personalized soundscapes. These soundscapes are aimed at improving focus, enhancing productivity, and providing a more pleasant auditory environment.
            </p>
            <p>
                Among its core features, the application includes a <span className="highlight">Sound Customizer</span>, which allows users to adjust various sound profiles according to their preferences and requirements. Additionally, the <span className="highlight">AI Personalizer</span> component uses intelligent algorithms to adapt the sound experience to individual user behaviors and environmental conditions. Furthermore, the <span className="highlight">Adaptive Sound Masking Module</span> dynamically responds to real-time changes in environmental noise, ensuring that users always experience optimal sound masking performance.
            </p>
            <p>
                The development of this project has been driven by a commitment to simplicity and ease of use. The frontend is built using <span className="highlight">React.js</span>, which provides a responsive and intuitive user interface. On the backend, <span className="highlight">Supabase</span> is utilized for managing data and authentication, offering a robust and scalable solution for backend operations. Meanwhile, <span className="highlight">Python</span> is employed for creating and fine-tuning machine learning models that drive the application's advanced features. Together, these technologies ensure a seamless and efficient user experience, combining powerful backend capabilities with an engaging and user-friendly frontend.

            </p>
        </div>
    );
};

export default About;

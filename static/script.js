

       
document.addEventListener('DOMContentLoaded', function () {
    const askButton = document.getElementById('ask-question-btn');
    const resultBox = document.getElementById('result-box');
    const loaderContainer = document.getElementById('loader-container');
    const selectedFile = document.getElementById('selected_file');
    // const selectedLanguage = document.getElementById('selected_language');
    const queryText = document.getElementById('query_text');
    const docSelectionMessage = document.getElementById('doc-selection-message');

    // Convert text into hyperlinks
    function convertTextToHyperlinks(text) {
        if (text.includes(':')) {
            let [leftPart, rightPart] = text.split(':');
            let links = rightPart.split(',').map(item => 
                `<a href="#" class="query-link">${item.trim()}</a>`).join(', ');
            return `<strong>${leftPart}:</strong> ${links}`;
        }
        return text;
    }

    // Add event listener for dynamically generated links
    document.getElementById("chatbox-container").addEventListener("click", function (e) {
        if (e.target.classList.contains("query-link")) {
            e.preventDefault();
            document.getElementById("query_text").value = e.target.textContent.trim();
            if (document.getElementById("query_text").value !== "") {
                document.getElementById("ask-question-btn").click();
            }
        }
    });

    document.getElementById('clearBtn').addEventListener('click', function() {
        // Send a POST request to the '/clear' route
        fetch('/clear', {
            method: 'POST',  // Use POST method for the request
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => {
            // Once the request is successful, reload the chat page
            if (response.ok) {
                window.location.href = 'chatpage';  // Redirect to the chat page
            } else {
                console.error('Failed to clear the data');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    
    function updateButtonVisibility() {
        askButton.style.display = (selectedFile.value !== '') ? 'block' : 'none';
    }

    // Convert URLs in text to clickable hyperlinks
    function convertTextToHyperlinks(text) {
        const urlPattern = /(https?:\/\/[^\s]+)/g; // Regex to detect URLs
        return text.replace(urlPattern, (url) => {
            return `<a href="${url}" target="_blank">${url}</a>`;
        });
    }

    // Function to handle sending data to answers.json
    function sendToAnswersJson(prompt, chatAns, humanAns) {
        const answerData = {
            Prompt: prompt,
            "Chat-Ans": chatAns,
            "Human-Ans": humanAns
        };

        fetch('/save_answers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Answer data saved:', data);
        })
        .catch(error => {
            console.error('Error saving answer data:', error);
        });
    }

    askButton.addEventListener('click', function () {
        if (selectedFile.value === '') {
            docSelectionMessage.textContent = 'Please select a document.';
            return;
        } else {
            docSelectionMessage.textContent = ''; // Clear message if valid
        }

        // Show loader and hide ask button
        loaderContainer.style.display = 'flex';
        askButton.style.display = 'none';

        const formData = new FormData();
        formData.append('query_text', queryText.value);
        formData.append('selected_file', selectedFile.value);
        // formData.append('selected_language', selectedLanguage.value);

        fetch('/ask', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const question = queryText.value;
          //  const answer = convertTextToHyperlinks(data.answer);
              const answer = data.answer.replace(/\n/g, "<br>");

            // Show question in chatbox
            resultBox.innerHTML += `
                <div class="message-block">
              <div class="message user"><strong>Ques: </strong>${question}</div>
              <div class="message bot"><strong>Ans: </strong>${answer}</div>
              <div class="feedback-section">
                  <div class="feedback-icons">

                      <span class="feedback correc">&#128077;</span> <!-- Thumbs Up -->
                      <span class="feedback wrong">&#128078;</span> <!-- Thumbs Down -->

                  </div>
                  <div class="feedback-box" style="display: none;">
                      <div class="feedback-input">
                          <textarea class="feedback-text" placeholder="Please provide your feedback..."></textarea>
                          <button class="submit-feedback">Submit</button>
                      </div>
                      <span class="feedback-error" style="color:red; display:none;">Please provide feedback.</span>
                  </div>
              </div>
          </div>`;

            // Scroll to the bottom of the chatbox
            resultBox.scrollTop = resultBox.scrollHeight;

            // Hide loader and show button again
            loaderContainer.style.display = 'none';
            askButton.style.display = 'inline-block';
    

// Send answer data to answers.json if user clicks correct or wrong
// Send answer data to answers.json if user clicks correct or wrong
document.querySelectorAll('.feedback.correc').forEach((correctButton, index) => {
  correctButton.addEventListener('click', function () {
      // Save correct answer in JSON
      const correctQuestion = document.querySelectorAll('.message.user')[index]?.textContent || "Unknown Question";
      const correctAnswer = document.querySelectorAll('.message.bot')[index]?.textContent || "Unknown Answer";

      sendToAnswersJson(correctQuestion, correctAnswer, ""); // Human answer is empty for correct
      saveFeedback(correctQuestion, correctAnswer, true);

      // Show "Thank you" popup for thumbs-up click
      showSuccessPopup("Thank you for your response!");
  });
});

// Function to display the success popup message
function showSuccessPopup(message) {
  const feedbackPopup = document.createElement('div');
  feedbackPopup.classList.add('feedback-popup');
  feedbackPopup.style.position = 'fixed';
  feedbackPopup.style.top = '50%';
  feedbackPopup.style.left = '50%';
  feedbackPopup.style.transform = 'translate(-50%, -50%)';
  feedbackPopup.style.backgroundColor = '#96DFF7';
  feedbackPopup.style.color = 'black';
  feedbackPopup.style.padding = '20px';
  feedbackPopup.style.borderRadius = '5px';
  feedbackPopup.style.zIndex = '1000';
  feedbackPopup.textContent = message;

  document.body.appendChild(feedbackPopup);

  // Hide popup after 3 seconds
  setTimeout(function () {
      feedbackPopup.style.display = 'none';
      document.body.removeChild(feedbackPopup);
  }, 3000);
}
document.querySelectorAll('.feedback.wrong').forEach((wrongButton, index) => {
  wrongButton.addEventListener('click', function () {
      // Show feedback box for wrong answers
      const feedbackBox = document.querySelectorAll('.feedback-box')[index];
      feedbackBox.style.display = 'block';

      // Handle submit feedback
      const submitFeedbackButton = feedbackBox.querySelector('.submit-feedback');
      const submitFeedbackHandler = function () {
          const feedbackText = feedbackBox.querySelector('.feedback-text').value;
          const feedbackError = feedbackBox.querySelector('.feedback-error');

          // Check if feedback is empty
          if (feedbackText.trim() === '') {
              feedbackError.style.display = 'block'; // Show error message
          } else {
              feedbackError.style.display = 'none'; // Hide error message

              const wrongQuestion = document.querySelectorAll('.message.user')[index].textContent;
              const wrongAnswer = document.querySelectorAll('.message.bot')[index].textContent;

              sendToAnswersJson(wrongQuestion, wrongAnswer, feedbackText); // Include feedback
              saveFeedback(wrongQuestion, wrongAnswer, false, feedbackText);

              // Hide the feedback box after submission
              feedbackBox.style.display = 'none';

              // Display "Your feedback has been submitted successfully!" popup for wrong answer feedback
              showSuccessPopup("Your feedback has been submitted successfully!");

              // Remove the event listener to prevent multiple submissions for the same feedback
              submitFeedbackButton.removeEventListener('click', submitFeedbackHandler);
          }
      };

      submitFeedbackButton.addEventListener('click', submitFeedbackHandler);
  });
});

    // Clear the input text area after sending the question
            queryText.value = ''; 
        })
        .catch(error => {
            console.error('Error:', error);
        })
        .finally(() => {
            loaderContainer.style.display = 'none'; // Hide loader
            askButton.style.display = 'block'; // Show ask button again
        });
    });

    // Handle Enter and Shift+Enter in the textarea
    queryText.addEventListener('keypress', function (event) {
        if (event.key === 'Enter') {
            if (event.shiftKey) {
                // Allow new line
                return;
            } else {
                // Trigger ask button click
                askButton.click();
                event.preventDefault(); // Prevent new line
            }
        }
    });

    selectedFile.addEventListener('change', function () {
        if (selectedFile.value === '') {
            docSelectionMessage.textContent = 'Please select a document.';
        } else {
            docSelectionMessage.textContent = '';
        }
        updateButtonVisibility();
    });

    // selectedLanguage.addEventListener('change', function () {
    //     updateButtonVisibility();
    // });

    // Initialize button visibility on page load
    updateButtonVisibility();



    // Function to save feedback in the database with selected file name
function saveFeedback(question, answer, isCorrect, feedback = '') {
    const selectedFile = document.querySelector('#selected_file').value; // Get selected file

    

    const feedbackData = {
        selected_file: selectedFile, // Include the selected file name
        question: question,
        answer: answer,
        isCorrect: isCorrect,
        feedback: feedback
    };

    fetch('/save_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            console.log('Feedback saved:', data);

        }
    })
    .catch(error => {
        console.error('Error saving feedback:', error);
    });
}


});




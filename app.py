import pandas as pd
import streamlit as st
import requests

# Create a Streamlit app
st.title("Toxicity Detector")
st.write("Enter a comment to detect its toxicity:")

# Create a text input field with a fun font
# comment = st.text_input(
#     "Comment", height=200, placeholder="Type something toxic...",)
comment = st.text_area("Comment", height=100,
                       placeholder="Type something toxic...")

# Create a button to trigger the prediction
if st.button("Detect"):
    # Send a POST request to your FastAPI endpoint
    response = requests.post(
        "http://localhost:8000/predict", json={"text": comment})

    # Parse the response as JSON
    output = response.json()

    # Create a pandas dataframe to display the results
    df = pd.DataFrame({"Category": list(output.keys()),
                      "Toxicity": list(output.values())})

    # Display the results in a table with a fun theme
    st.write("Toxicity Report:")
    st.table(df.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "black"}))

    # Add some fun emojis to the report
    if any(value == 1 for value in output.values()):
        st.write("😬 Oh no! Your comment is toxic! 🚫")
    else:
        st.write("🎉 Yay! Your comment is not toxic! 👍")

# Add some fun features to the app
st.write("Want to test your toxicity skills? 🤔")
st.write("Try typing a toxic comment and see how well our model detects it! 😈")

# Add a footer with a fun message
st.write("Made with ❤️ by [muhammadof9@gmail.com]")

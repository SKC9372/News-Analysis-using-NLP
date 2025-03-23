import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Company News Analysis and Sentiment Analysis")

company_name = st.text_input("Enter Company Name",placeholder="For example: Apple, Google, etc.")

if st.button("Run Analysis"):
    with st.spinner("Fetching News..."):
        response = requests.post(f"{API_URL}/fetch_news", json={"company_name":company_name})
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles",[])
            report = data.get("report",{})

        # Display Articles
            st.subheader("📰 Fetched Articles")
            for article in articles:
                st.markdown(f"**{article['title']}**")
                st.write(f"📌 Sentiment: {article['sentiment']}")
                st.write(f"📌 Summary: {article['summary']}")
                st.write(f"📌 Topics: {', '.join(article['topics'])}")
                st.write(f"🔗 [Read More]({article['url']})")
                st.write("---")   
        
        # Display Sentiment Distribution
            st.subheader("📊 Sentiment Analysis")
            sentiment_counts = report.get("Sentiment Distribution", {})
            
            st.bar_chart(sentiment_counts)

            # Display Final Summary
            st.subheader("📄 Final Summary")
            st.write(report.get("Final Sentiment Analysis", "No summary available."))

            # ✅ Fetch & Play the Audio File
            st.subheader("🔊 Hindi Audio Summary")
            audio_url = f"{API_URL}/download-audio"
            
            # Download audio file
            audio_response = requests.get(audio_url)
            if audio_response.status_code == 200:
                with open("summary.mp3", "wb") as f:
                    f.write(audio_response.content)  # Save the file
                
                st.audio("summary.mp3", format="audio/mp3")  # Play saved file
            else:
                st.error("❌ Error: Could not fetch the audio file.")


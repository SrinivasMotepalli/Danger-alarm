import streamlit as st

def main():
    # Apply CSS styling to create a gradient background
    st.markdown(
        """
        <style>
        body {
            background-image: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-size: cover;
            padding: 20px; /* Add padding to ensure gradient covers entire app */
        }

        .title {
            color: #333333;
            text-align: center;
            font-size: 36px;
        }

        .group-members {
            color: #666666;
            font-size: 24px;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<h1 class='title'>Danger Detection</h1>", unsafe_allow_html=True)

    # Group members list
    st.markdown("<p class='group-members'>Group Members:</p>", unsafe_allow_html=True)
    st.markdown("<ul class='group-members'>", unsafe_allow_html=True)
    st.markdown("<li>Drishtti Narwal</li>", unsafe_allow_html=True)
    st.markdown("<li>Srinivas Motipalli</li>", unsafe_allow_html=True)
    st.markdown("<li>Satya Prem</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

if _name_ == "_main_":
    main()

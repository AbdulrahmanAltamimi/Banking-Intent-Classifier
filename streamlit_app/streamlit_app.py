import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Page configuration
st.set_page_config(
    page_title="Banking Intent Classifier",
    layout="centered"
)

# Intent labels (77 banking intents)
INTENT_LABELS = [
    "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
]

MODEL_ID = "abdulrahmanMoh/bert_Banking77"

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer from Hugging Face Hub."""
    model = BertForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

def predict_intent(text, model, tokenizer, device, top_k=5):
    """Predict intent for given text."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'intent': INTENT_LABELS[idx],
            'confidence': float(prob)
        })

    return results

def main():
    st.title("Banking Intent Classifier")
    st.markdown("Classify customer queries into 77 banking intents using BERT")

    st.markdown("---")

    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()

    device_info = "GPU" if device.type == 'cuda' else "CPU"
    st.caption(f"Running on: {device_info}")

    # Initialize session state for text input
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    # Input section
    st.subheader("Enter your query")

    # Example queries
    examples = [
        "I want to transfer money to another account",
        "My card was declined at the store",
        "How do I change my PIN number?",
        "I made a transaction but did it to the wrong account."
    ]

    # Callback function to set example text
    def set_example(example_text):
        st.session_state.text_input = example_text

    # Quick example buttons
    st.markdown("**Try an example:**")
    cols = st.columns(4)
    for i, example in enumerate(examples[:4]):
        cols[i].button(
            f"Example {i+1}",
            key=f"ex_{i}",
            on_click=set_example,
            args=(example,)
        )

    # Text input with session state key
    user_input = st.text_area(
        "Type your banking query here:",
        height=100,
        placeholder="e.g., I want to check my account balance",
        key="text_input"
    )

    # Predict button
    if st.button("Classify Intent", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                results = predict_intent(user_input, model, tokenizer, device)

            st.markdown("---")
            st.subheader("Prediction Results")

            # Top prediction
            top_result = results[0]
            st.success(f"**Predicted Intent:** {top_result['intent'].replace('_', ' ').title()}")
            st.metric("Confidence", f"{top_result['confidence']:.1%}")

            # Show top 5 predictions
            st.markdown("**Top 5 Predictions:**")
            for i, result in enumerate(results):
                intent_display = result['intent'].replace('_', ' ').title()
                confidence = result['confidence']

                st.write(f"**{i+1}. {intent_display}** - {confidence:.1%}")
                st.progress(confidence)
        else:
            st.warning("Please enter a query to classify.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with Streamlit and Hugging Face Transformers</p>
            <p>Model: BERT fine-tuned on Banking77 dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

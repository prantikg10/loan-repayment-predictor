import streamlit as st
import main
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Repayment Predictor")
st.sidebar.title("Loan Repayment Predictor")


model = st.sidebar.selectbox("Select Model:", [' ', 'Decision Tree', 'Random Forest', 'Gradient Boosting'])
btn = st.sidebar.button("Train Model")

if btn:
    if model == 'Decision Tree':

        grid, y_pred_test, train_accuracy, test_accuracy = main.train_decision_tree(
            main.X_train,
            main.X_test,
            main.y_train,
            main.y_test
        )
        st.success("Decision Tree training completed!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Best Parameter")
            st.write(grid.best_params_)
        with col2:
            st.subheader("Best Score")
            st.write(grid.best_score_)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Train Accuracy")
            st.write(train_accuracy)
        with col4:
            st.subheader("Test Accuracy")
            st.write(test_accuracy)

        if (train_accuracy - test_accuracy) > 0.08:
            st.warning("Model is overfitting")
        else:
            st.success("Model looks good")

    elif model == 'Random Forest':

        rf_clf, y_pred_test, train_accuracy, test_accuracy = main.train_random_forest(
            main.X_train,
            main.X_test,
            main.y_train,
            main.y_test
        )
        st.success("Random Forest training completed!")

        st.subheader("Best Estimator:")
        st.write(rf_clf.n_estimators)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Train Accuracy")
            st.write(train_accuracy)
        with col2:
            st.subheader("Test Accuracy")
            st.write(test_accuracy)

        if (train_accuracy - test_accuracy) > 0.08:
            st.warning("Model is overfitting")
        else:
            st.success("Model looks good")

    elif model == 'Gradient Boosting':

        gb_clf, y_pred_test, train_accuracy, test_accuracy = main.train_gradient_boosting(
            main.X_train,
            main.X_test,
            main.y_train,
            main.y_test
        )
        st.success("Gradient Boosting training completed!")

        st.subheader("Learning rate used:")
        st.write(gb_clf.learning_rate)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Train Accuracy")
            st.write(train_accuracy)
        with col2:
            st.subheader("Test Accuracy")
            st.write(test_accuracy)

        if (train_accuracy - test_accuracy) > 0.08:
            st.warning("Model is overfitting")
        else:
            st.success("Model looks good")

    else:
        st.error("Please select an option from the dropdown to continue.")


option = st.sidebar.selectbox("Select a Visualization:", [' ', 'Credit Score vs. Credit Policy', 'Credit Score vs. Loan Repayment', 'Correlation Between Numerical Features'])
btn1 = st.sidebar.button("Visualize")

if btn1:

    if option == 'Credit Score vs. Credit Policy':

        st.subheader("Credit Score vs. Credit Policy")

        plt.figure(figsize=(8, 5))

        plt.hist(
        main.loan['credit_score'].loc[main.loan['credit_policy'] == 1],
        bins=30,
        label='Credit_Policy = 1'
        )

        plt.hist(
        main.loan['credit_score'].loc[main.loan['credit_policy'] == 0],
        bins=30,
        label='Credit_Policy = 0'
        )

        plt.legend()
        plt.xlabel('Credit Score')

        st.pyplot(plt)

    elif option == 'Credit Score vs. Loan Repayment':

        st.subheader("Credit Score Distribution vs. Loan Repayment")

        plt.figure(figsize=(10, 6))

        main.loan[main.loan['not_fully_paid'] == 1]['credit_score'].hist(bins=30, alpha=0.5, color='blue',
                                                               label='not_fully_paid = 1')
        main.loan[main.loan['not_fully_paid'] == 0]['credit_score'].hist(bins=30, alpha=0.5, color='green',
                                                               label='not_fully_paid = 0')
        plt.legend()
        plt.xlabel('credit_score')

        st.pyplot(plt)

    elif option == 'Correlation Between Numerical Features':
        st.subheader("Heatmap - Correlation Between Numerical Features")
        plt.figure(figsize=(20, 15))
        sns.heatmap(main.loan.corr(), cmap='YlGnBu', annot=True)

        st.pyplot(plt)

    else:
        st.error("Please select an option from the dropdown to continue.")


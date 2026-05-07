package com.example.ultimatestocks

import android.annotation.SuppressLint
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.VerticalAlignmentLine
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.CurrentScreen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.ui.theme.UltimateStocksTheme

val authLightBlue = Color(0xFFE3F2FD)
val authAccentBlue = Color(0xFF2196F3)
val authDeepBlue = Color(0xFF1976D2)
val authSoftBlue = Color(0xFFBBDEFB)
val authPrimaryBlack = Color(0xFF111111)
val authSecondaryBlack = Color(0xFF2D2D2D)
val authPureWhite = Color(0xFFFFFFFF)

class AuthPage : Screen {
    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    @Composable
    override fun Content() {
        UltimateStocksTheme {
            Navigator(WelcomeScreen()) { navigator ->
                Scaffold(
                    modifier = Modifier.fillMaxSize()
                        .background(authPureWhite),
                    content = {
                        Column(
                            modifier = Modifier.padding(top = 30.dp)
                        ) {
                            CurrentScreen()
                        }
                    }
                )
            }
        }
    }
}

@Composable
fun AuthHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "USX",
            fontWeight = FontWeight.Black,
            fontSize = 25.sp,
            color = authDeepBlue
        )
        Text(
            modifier = Modifier
                .padding(horizontal = 10.dp)
                .align(Alignment.CenterVertically),
            text = "Ultimate Stock Exchange",
            fontSize = 14.sp,
            color = authPrimaryBlack
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
class LoginScreen : Screen {
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        var emailInput by remember { mutableStateOf("") }
        var passwordInput by remember { mutableStateOf("") }
        var errorMessage by remember { mutableStateOf("") } // Add error message state

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(authPureWhite)
                .padding(horizontal = 16.dp)
        ) {
            AuthHeader()
            Text(
                text = "SIGN IN",
                fontWeight = FontWeight.Black,
                fontSize = 40.sp,
                color = authDeepBlue,
                modifier = Modifier
                    .padding(top = 150.dp, bottom = 30.dp)
            )

            OutlinedTextField(
                value = emailInput,
                onValueChange = { emailInput = it },
                label = { Text("EMAIL", color = authSecondaryBlack) },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 20.dp),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email, imeAction = ImeAction.Done),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = authAccentBlue,
                    unfocusedBorderColor = authSoftBlue
                ),
                shape = RoundedCornerShape(15.dp)
            )

            OutlinedTextField(
                value = passwordInput,
                onValueChange = { passwordInput = it },
                label = { Text("PASSWORD", color = authSecondaryBlack) },
                modifier = Modifier.fillMaxWidth(),
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = authAccentBlue,
                    unfocusedBorderColor = authSoftBlue
                ),
                shape = RoundedCornerShape(15.dp)
            )

            // Show the error message if present
            if (errorMessage.isNotEmpty()) {
                Text(
                    text = errorMessage,
                    color = Color.Red,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }

            Button(
                onClick = {
                    MainActivity.model.attemptSignIn(
                        emailInput,
                        passwordInput,
                        onSuccess = { navigator.push(BottomNav()) },
                        onError = { error -> errorMessage = error } // Set the error message
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 20.dp, bottom = 10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = authAccentBlue),
                elevation = ButtonDefaults.buttonElevation(8.dp),
                shape = RoundedCornerShape(15.dp)
            ) {
                Text(
                    text = "Login",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = authPureWhite,
                    modifier = Modifier
                        .padding(vertical = 10.dp)
                        .fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }

            Column(
                modifier = Modifier.fillMaxSize()
                    .padding(50.dp),
                verticalArrangement = Arrangement.Bottom
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "Don't have an account? ",
                        textAlign = TextAlign.Center,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = authPrimaryBlack,
                    )
                    Text(
                        text = "Sign Up",
                        textAlign = TextAlign.Center,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = authDeepBlue,
                        modifier = Modifier
                            .clickable { navigator.push(SignupScreen()) }
                    )
                }
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
class SignupScreen : Screen {
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        var emailInput by remember { mutableStateOf("") }
        var userName by remember { mutableStateOf("") }
        var passwordInput by remember { mutableStateOf("") }
        var confirmPasswordInput by remember { mutableStateOf("") }
        var errorMessage by remember { mutableStateOf("") } // Add error message state

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(authPureWhite)
                .padding(horizontal = 16.dp)
        ) {
            AuthHeader()

            Column(
                modifier = Modifier
                    .fillMaxWidth()
            ) {
                Text(
                    text = "SIGN UP",
                    fontWeight = FontWeight.Black,
                    fontSize = 40.sp,
                    color = authDeepBlue,
                    modifier = Modifier
                        .padding(top = 30.dp, bottom = 30.dp)
                )

                OutlinedTextField(
                    value = emailInput,
                    onValueChange = { emailInput = it },
                    label = { Text("EMAIL", color = authSecondaryBlack) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 20.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email, imeAction = ImeAction.Done),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = authAccentBlue,
                        unfocusedBorderColor = authSoftBlue
                    ),
                    shape = RoundedCornerShape(15.dp)
                )

                OutlinedTextField(
                    value = userName,
                    onValueChange = { userName = it },
                    label = { Text("USERNAME", color = authSecondaryBlack) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 20.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text, imeAction = ImeAction.Done),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = authAccentBlue,
                        unfocusedBorderColor = authSoftBlue
                    ),
                    shape = RoundedCornerShape(15.dp)
                )

                OutlinedTextField(
                    value = passwordInput,
                    onValueChange = { passwordInput = it },
                    label = { Text("PASSWORD", color = authSecondaryBlack) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 20.dp),
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = authAccentBlue,
                        unfocusedBorderColor = authSoftBlue
                    ),
                    shape = RoundedCornerShape(15.dp)
                )

                OutlinedTextField(
                    value = confirmPasswordInput,
                    onValueChange = { confirmPasswordInput = it },
                    label = { Text("CONFIRM PASSWORD", color = authSecondaryBlack) },
                    modifier = Modifier.fillMaxWidth(),
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = authAccentBlue,
                        unfocusedBorderColor = authSoftBlue
                    ),
                    shape = RoundedCornerShape(15.dp)
                )

                // Show the error message if present
                if (errorMessage.isNotEmpty()) {
                    Text(
                        text = errorMessage,
                        color = Color.Red,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }

                Button(
                    onClick = {
                        // Validate input before attempting to sign up
                        if (emailInput.isBlank()) {
                            errorMessage = "Email cannot be empty."
                            return@Button
                        }
                        if (userName.isBlank()) {
                            errorMessage = "Username cannot be empty."
                            return@Button
                        }
                        if (passwordInput.isBlank()) {
                            errorMessage = "Password cannot be empty."
                            return@Button
                        }
                        if (confirmPasswordInput.isBlank()) {
                            errorMessage = "Please confirm your password."
                            return@Button
                        }
                        if (passwordInput != confirmPasswordInput) {
                            errorMessage = "Passwords do not match."
                            return@Button
                        }
                        MainActivity.model.attemptSignUp(
                            emailInput,
                            passwordInput,
                            confirmPasswordInput,
                            userName,
                            onSuccess = { navigator.push(BottomNav()) },
                            onError = { error -> errorMessage = error } // Set the error message
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 20.dp, bottom = 10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = authAccentBlue),
                    elevation = ButtonDefaults.buttonElevation(8.dp),
                    shape = RoundedCornerShape(15.dp)
                ) {
                    Text(
                        text = "Sign Up",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = authPureWhite,
                        modifier = Modifier
                            .padding(vertical = 10.dp)
                            .fillMaxWidth(),
                        textAlign = TextAlign.Center
                    )
                }
            }

            Column(
                modifier = Modifier.fillMaxSize()
                    .padding(50.dp),
                verticalArrangement = Arrangement.Bottom
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "Have an account? ",
                        textAlign = TextAlign.Center,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = authPrimaryBlack,
                    )
                    Text(
                        text = "Log In",
                        textAlign = TextAlign.Center,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = authDeepBlue,
                        modifier = Modifier
                            .clickable { navigator.push(LoginScreen()) }
                    )
                }
            }
        }
    }
}


class WelcomeScreen : Screen {
    @Composable
    override fun Content() {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(authPureWhite),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "USX",
                fontSize = 80.sp,
                fontWeight = FontWeight.Bold,
                color = authDeepBlue,
                modifier = Modifier.padding(top = 100.dp)
            )
            Text(
                text = "Ultimate Stock Exchange",
                fontSize = 24.sp,
                color = authPrimaryBlack
            )
            Text(
                text = "WELCOME TO USX",
                textAlign = TextAlign.Center,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = authDeepBlue,
                modifier = Modifier
                    .width(200.dp)
                    .padding(top = 140.dp)
            )
            Column(
                modifier = Modifier.padding(top = 100.dp)
            ) {
                LoginButton()
                SignupButton()
            }
        }
    }
}

class ProtectedScreen : Screen {
    @Composable
    override fun Content() {
        var navigator = LocalNavigator.currentOrThrow
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(authPureWhite)
                .padding(16.dp)
        ) {
            Text(
                text = "User Logged In: " + MainActivity.model.auth.currentUser?.email,
                color = authPrimaryBlack
            )
            Button(
                onClick = {
                    MainActivity.model.attemptLogout()
                    navigator.push(WelcomeScreen())
                },
                colors = ButtonDefaults.buttonColors(containerColor = authAccentBlue),
                elevation = ButtonDefaults.buttonElevation(8.dp),
                shape = RoundedCornerShape(15.dp)
            ) {
                Text(
                    text = "Log Out",
                    color = authPureWhite,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun LoginButton() {
    val navigator = LocalNavigator.currentOrThrow
    Button(
        onClick = { navigator.push(LoginScreen()) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 10.dp),
        colors = ButtonDefaults.buttonColors(containerColor = authAccentBlue),
        elevation = ButtonDefaults.buttonElevation(8.dp),
        shape = RoundedCornerShape(15.dp)
    ) {
        Text(
            text = "Login",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = authPureWhite,
            modifier = Modifier.padding(vertical = 10.dp),
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun SignupButton() {
    val navigator = LocalNavigator.currentOrThrow
    OutlinedButton(
        onClick = { navigator.push(SignupScreen()) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp),
        colors = ButtonDefaults.outlinedButtonColors(
            containerColor = authPureWhite,
            contentColor = authAccentBlue
        ),
        border = ButtonDefaults.outlinedButtonBorder.copy(width = 2.dp),
        shape = RoundedCornerShape(15.dp)
    ) {
        Text(
            text = "Sign Up",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(vertical = 10.dp),
            textAlign = TextAlign.Center
        )
    }
}
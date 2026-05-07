package com.example.ultimatestocks

//import androidx.compose.foundation.layout.FlowColumnScopeInstance.align
//import androidx.compose.foundation.layout.FlowRowScopeInstance.align
import android.annotation.SuppressLint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import cafe.adriel.voyager.navigator.Navigator
import com.example.ultimatestocks.model.Model
import com.example.ultimatestocks.view.ViewModel
import com.google.firebase.Firebase
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.auth


//lateinit var auth: FirebaseAuth


class MainActivity : ComponentActivity() {
    companion object {
        val model = Model()
        val viewModel = ViewModel(model)
    }

    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//        auth = Firebase.auth
        enableEdgeToEdge()
        setContent {
            if (model.auth.currentUser !== null) Navigator(listOf(AuthPage(), BottomNav()), onBackPressed = { false }) else Navigator(AuthPage(), onBackPressed = { false })

        }
    }
}

package com.example.ultimatestocks.view

import android.icu.util.Currency
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.model.ISubscriber
import com.example.ultimatestocks.model.Model

// The ViewModel contains all of the data that the View (aka the Composable function) displays.
class ViewModel(val model: Model) : ISubscriber {
    val sandboxes = mutableStateListOf<Sandbox>()

    var userUID = mutableStateOf("")
    var userEmail = mutableStateOf("")
    var userName = mutableStateOf("")

    var Currency = mutableStateOf("")
    var BuySellState = mutableStateOf("")

    var competitions = mutableStateListOf<Competition>()

    init {
        model.subscribe(this)
        update()
    }


    override fun update() {
        sandboxes.clear()
        for (sandbox in model.sandboxes) {
            sandboxes.add(sandbox)
        }
        userUID.value = model.userUID
        userEmail.value = model.userEmail
        userName.value = model.userName

        Currency.value = model.Currency
        BuySellState.value = model.BuySellState

        competitions.clear()
        for (comp in model.activeComps) {
            competitions.add(comp)
        }
    }
}
package com.example.ultimatestocks

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ultimatestocks.entities.StockInventory


@Composable
fun FinancialInfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = 16.sp,
            color = secondaryBlackk
        )
        Text(
            text = value,
            fontSize = 16.sp,
            color = pprimaryBlack,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun MyCustomTextField(searchQuery: String, onQueryChanged: (String) -> Unit) {
    OutlinedTextField(
        value = searchQuery,
        onValueChange = { onQueryChanged(it) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        label = { Text("Search", color = secondaryBlackk) },
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = aaccentBlue,
            unfocusedBorderColor = ssoftBlue
        ),
        shape = RoundedCornerShape(15.dp)
    )
}


@Composable
fun StockList(stocks: List<StockInventory>) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp)
    ) {
        items(stocks) { stock ->
            StockCard(stock)
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}


@Composable
fun StockCard(stock: StockInventory) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp, horizontal = 8.dp),
        elevation = CardDefaults.cardElevation(4.dp),
        colors = CardDefaults.cardColors(containerColor = llightBlue),
        shape = RoundedCornerShape(15.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = stock.ticker,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    color = pprimaryBlack
                )
            }
            Text(
                text = stock.quantityOwned.toString(),
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                color = pprimaryBlack
            )
        }
    }
}

package com.example.ultimatestocks.entities

import kotlinx.serialization.Serializable

@Serializable
data class StockNewPrice(
    val ticker: String,
    val newPrice: Float,
    val changeInPrice: Float
)

@Serializable
data class StockDetails(
    val ticker: String,
    val description: String, // short description of the company/stock so ChatGPT can infer how its impacted by a news event
    var historicPrice: MutableList<Float>, // historic and current price/share data
)

@Serializable
data class StockInventory(
    val ticker: String,
    var quantityOwned: Int,
)

data class FirebaseStock(
    val companyName: String,
    val description: String,
    val sector: String,
    val startingPrice: Double,
    val ticker: String
)

//@Serializable
//data class StocksOwned(
//    val companyName: String,
//    val tickerSymbol: String,
//    val sharesOwned: Int,
//    val pricePerShare: Double
//)

@Serializable
data class Stock(
    val ticker: String,
    val companyName: String,
    val description: String,
    val startingPrice: Float,
    val sector: String
)


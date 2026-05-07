package com.example.ultimatestocks.entities

import kotlinx.serialization.Serializable
import kotlin.time.Duration

// A data class for a Sandbox
@Serializable
data class Sandbox(
    val id: String = java.util.UUID.randomUUID().toString(),
    val name: String,
//    val timeDuration: String, // not relevant for now
    var newsEvents: MutableList<NewsEvent>,
    var latestNewsNotRevealed: String,
    val allStocks: List<StockDetails>,
    val ownedStocks: MutableList<StockInventory>,
    val portfolioVal: MutableList<Float>,
    var cash: Float
) {
    fun getStockValue(target: String): Float {
        for (stock in allStocks) {
            if (stock.ticker == target) return stock.historicPrice.last()
        }
        return 0.0f
    }
    fun buy(ticker: String, numberOfShares: Int) {
        val price = getStockValue(ticker)
        cash -= (price * numberOfShares)
        var added = false
        for (stock in ownedStocks) {
            if (stock.ticker == ticker) {
                stock.quantityOwned += numberOfShares
                added = true
            }
        }
        if (!added) ownedStocks.add(StockInventory(ticker, numberOfShares))
    }

    fun sell(ticker: String, numberOfShares: Int) {
        val goingPrice = getStockValue(ticker)
        cash += goingPrice * numberOfShares
        for (stock in ownedStocks) {
            if (stock.ticker == ticker) {
                if (stock.quantityOwned > numberOfShares) {
                    stock.quantityOwned -= numberOfShares
                } else {
                    ownedStocks.remove(stock)
                }
                break
            }
        }
    }
}
//{
//    fun getPortfolioValue(): Double {
//        return ownedStocks.fold(0.0) {acc, element -> acc + (element.sharesOwned * element.pricePerShare)}
//    }
//}

@Serializable
data class NewsEvent(
    val title: String,
    val body: String,
    val newPrices: List<StockNewPrice>
)





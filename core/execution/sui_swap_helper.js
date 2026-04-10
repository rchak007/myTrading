#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * Handles SUI's UTXO-like coin model by merging all coin objects of the
 * input type before swapping — same as what Slush wallet does automatically.
 *
 * Setup (in core/execution/):
 *   npm install @7kprotocol/sdk-ts @mysten/sui@1 @flowx-finance/sdk @cetusprotocol/aggregator-sdk axios --legacy-peer-deps
 */

async function main() {
    const args = parseArgs(process.argv.slice(2));

    const {
        coinIn,
        coinOut,
        amount,
        wallet,
        privateKey,
        slippage = "0.03",
        rpcUrl,
    } = args;

    if (!coinIn || !coinOut || !amount || !wallet || !privateKey) {
        output({ status: "error", message: "Missing required args: --coinIn --coinOut --amount --wallet --privateKey" });
        process.exit(1);
    }

    try {
        const { SuiClient, getFullnodeUrl } = await import("@mysten/sui/client");
        const { Ed25519Keypair } = await import("@mysten/sui/keypairs/ed25519");
        const { Transaction } = await import("@mysten/sui/transactions");
        const sevenK = await import("@7kprotocol/sdk-ts");

        // Initialize SUI client
        const suiClient = new SuiClient({
            url: rpcUrl || getFullnodeUrl("mainnet"),
        });

        // Initialize MetaAg
        const ma = new sevenK.MetaAg({ suiClient });

        // Step 0: Merge all coins of the input type first
        // SUI uses UTXO model — tokens may be split across multiple coin objects
        output_log("Checking coin objects for merge...");
        await mergeCoinObjects(suiClient, wallet, coinIn, privateKey);

        // Step 1: Get quote
        output_log(`Getting quote: ${coinIn.split("::").pop()} -> ${coinOut.split("::").pop()} amount=${amount}`);

        const quotes = await ma.quote({
            coinTypeIn: coinIn,
            coinTypeOut: coinOut,
            amountIn: amount,
        });

        if (!quotes || quotes.length === 0) {
            throw new Error("No quotes returned — no liquidity providers available for this pair");
        }

        const bestQuote = quotes[0];
        const expectedOut = bestQuote.amountOut || bestQuote.rawAmountOut || "?";
        output_log(`Quote received: provider=${bestQuote.provider} expectedOut=${expectedOut}`);

        // Step 2: Build transaction via swap()
        output_log("Building swap transaction...");

        const tx = new Transaction();

        await ma.swap({
            signer: wallet,
            tx: tx,
            quote: bestQuote,
        });

        output_log("Transaction built, signing...");

        // Step 3: Parse private key and sign
        let keypair = parseKeypair(privateKey);

        tx.setSender(wallet);
        const txBytes = await tx.build({ client: suiClient });
        const { signature } = await keypair.signTransaction(txBytes);

        output_log("Executing transaction...");

        // Step 4: Execute
        const result = await suiClient.executeTransactionBlock({
            transactionBlock: txBytes,
            signature: [signature],
            options: {
                showEffects: true,
                showEvents: true,
            },
        });

        const digest = result.digest;
        const status = result.effects?.status?.status;

        if (status === "success") {
            output({
                status: "success",
                digest: digest,
                amountOut: String(expectedOut),
            });
        } else {
            const error = result.effects?.status?.error || "unknown";
            output({
                status: "error",
                message: `Transaction failed on-chain: ${error}`,
                digest: digest,
            });
            process.exit(1);
        }

    } catch (err) {
        output({
            status: "error",
            message: err.message || String(err),
            stack: err.stack ? err.stack.split("\n").slice(0, 5).join(" | ") : undefined,
        });
        process.exit(1);
    }
}


/**
 * Merge all coin objects of a given type into one.
 * This is what wallets like Slush do automatically before swaps.
 * Only merges if there are 2+ non-zero coin objects.
 */
async function mergeCoinObjects(suiClient, wallet, coinType, privateKey) {
    const { Transaction } = await import("@mysten/sui/transactions");

    // Fetch all coin objects of this type
    const allCoins = [];
    let cursor = null;
    while (true) {
        const page = await suiClient.getCoins({ owner: wallet, coinType, cursor });
        allCoins.push(...page.data);
        if (!page.hasNextPage) break;
        cursor = page.nextCursor;
    }

    // Filter to non-zero coins
    const nonZero = allCoins.filter(c => BigInt(c.balance) > 0n);

    output_log(`Found ${allCoins.length} coin objects (${nonZero.length} non-zero) for ${coinType.split("::").pop()}`);

    if (nonZero.length <= 1) {
        output_log("No merge needed — single coin object or empty");
        return;
    }

    // Merge all into the first coin
    output_log(`Merging ${nonZero.length} coin objects into one...`);

    const tx = new Transaction();
    const primary = nonZero[0];
    const toMerge = nonZero.slice(1);

    tx.mergeCoins(
        tx.object(primary.coinObjectId),
        toMerge.map(c => tx.object(c.coinObjectId)),
    );

    // Sign and execute the merge
    const keypair = parseKeypair(privateKey);
    tx.setSender(wallet);
    const txBytes = await tx.build({ client: suiClient });
    const { signature } = await keypair.signTransaction(txBytes);

    const result = await suiClient.executeTransactionBlock({
        transactionBlock: txBytes,
        signature: [signature],
        options: { showEffects: true },
    });

    const mergeStatus = result.effects?.status?.status;
    if (mergeStatus !== "success") {
        const err = result.effects?.status?.error || "unknown";
        throw new Error(`Coin merge failed: ${err}`);
    }

    output_log(`Merge successful: digest=${result.digest}`);

    // Wait a moment for the merge to be indexed
    await new Promise(r => setTimeout(r, 2000));
}


/**
 * Parse private key from hex or SUI bech32 format.
 */
function parseKeypair(privateKey) {
    if (privateKey.startsWith("suiprivkey")) {
        // Dynamic import for bech32 format
        throw new Error("suiprivkey format not yet supported — use hex format");
    }
    const { Ed25519Keypair } = require("@mysten/sui/keypairs/ed25519");
    const keyBytes = Buffer.from(privateKey.replace("0x", ""), "hex");
    return Ed25519Keypair.fromSecretKey(keyBytes);
}


function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith("--") && i + 1 < argv.length) {
            args[argv[i].slice(2)] = argv[i + 1];
            i++;
        }
    }
    return args;
}

function output(obj) {
    console.log(JSON.stringify(obj));
}

function output_log(msg) {
    process.stderr.write(`[sui_swap_helper] ${msg}\n`);
}

main();
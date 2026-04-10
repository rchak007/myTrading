#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * MetaAg.swap() expects:
 *   - signer: wallet address STRING (not keypair)
 *   - tx: a @mysten/sui Transaction object
 *   - quote: the quote result from MetaAg.quote()
 * It builds the swap commands into the Transaction, then we sign + execute.
 *
 * Setup (in core/execution/):
 *   npm install @7kprotocol/sdk-ts @mysten/sui@1 @flowx-finance/sdk axios --legacy-peer-deps
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
        // swap() expects: { signer: walletAddressString, tx: Transaction, quote: quoteResult }
        output_log("Building swap transaction...");

        const tx = new Transaction();

        await ma.swap({
            signer: wallet,        // wallet address as STRING
            tx: tx,                // Transaction object — swap() adds commands to it
            quote: bestQuote,
        });

        output_log("Transaction built, signing...");

        // Step 3: Parse private key and sign
        let keypair;
        if (privateKey.startsWith("suiprivkey")) {
            const { decodeSuiPrivateKey } = await import("@mysten/sui/cryptography");
            const parsed = decodeSuiPrivateKey(privateKey);
            keypair = Ed25519Keypair.fromSecretKey(parsed.secretKey);
        } else {
            const keyBytes = Buffer.from(privateKey.replace("0x", ""), "hex");
            keypair = Ed25519Keypair.fromSecretKey(keyBytes);
        }

        // Set sender and build
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
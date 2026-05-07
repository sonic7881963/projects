"use strict";
cc._RF.push(module, '5ceedp2BGlJTLjvk8lDqHNP', 'left');
// 3.16小游戏/command_TypeScript/level2/left.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var main_2 = require("./main");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (main_2.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                main_2.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                main_2.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                main_2.default.minion3_hp -= 30;
            }
        }
        main_2.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        main_2.default.attack = false;
        main_2.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((main_2.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight3");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(main_2.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(main_2.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(main_2.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (main_2.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();